"""Uji Stage 2: jalankan dari root proyek ->  python -m tests.test_stage2"""
import logging
import sys
from pathlib import Path

logging.disable(logging.WARNING)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from utils import config, data_loader as dl, model_loader as ml

ok_count = 0
def check(name, cond, extra=""):
    global ok_count
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    assert cond, name
    ok_count += 1

# 1. File penting tersedia
for p in [config.DATA_FILE, config.SCORED_FILE, config.BUNDLE_FILE, config.SCHEMA_FILE,
          config.MODELS_DIR / "customer_features.py"]:
    check(f"file ada: {p.relative_to(config.ROOT)}", p.exists())

# 2. Dataset bawaan lewat path relatif
raw = dl.load_raw_default()
check("dataset bawaan 7043 x 21", raw.shape == (7043, 21), str(raw.shape))

# 3. Tabel hasil notebook
scored = dl.load_scored_default()
check("tabel hasil notebook 7043 baris", len(scored) == 7043)
check("P1 = 434 (sama dengan priority_summary.csv)", int(scored["priority"].str.startswith("P1").sum()) == 434)

# 4. Model
b = ml.load_model_bundle()
check("bundle: XGBoost, kmeans, k=3", (b["model_name"], b["final_method"], b["k"]) == ("XGBoost", "kmeans", 3))
check("tidak ada peringatan versi library", b["_version_notes"] == [], str(b["_version_notes"]))

# 4b. Jalur cadangan portabel (simulasi bundle asli gagal dibaca, seperti di Windows)
import joblib as _jl
_orig_load = ml.joblib.load
def _fail_on_original(path, *a, **k):
    if Path(path) == config.BUNDLE_FILE:
        raise RuntimeError("simulasi: input stream corrupted")
    return _orig_load(path, *a, **k)
ml.load_model_bundle.clear()
ml.joblib.load = _fail_on_original
try:
    bp = ml.load_model_bundle()
    sc_p = ml.score_raw(raw.drop(columns=["Churn"]).head(300))
finally:
    ml.joblib.load = _orig_load
    ml.load_model_bundle.clear()
b = ml.load_model_bundle()
sc_o = ml.score_raw(raw.drop(columns=["Churn"]).head(300))
check("cadangan portabel dipakai saat bundle asli gagal", bp["_source"] == "portabel")
check("skor portabel identik dengan bundle asli",
      np.array_equal(sc_p["churn_prob_raw"].values, sc_o["churn_prob_raw"].values)
      and (sc_p["priority"].values == sc_o["priority"].values).all())

# 5. Validasi dataset valid
res = dl.validate_dataset(raw)
check("validasi dataset asli: lolos", res.ok, str(res.errors))
check("11 TotalCharges kosong dilaporkan", any("11 TotalCharges" in w for w in res.warnings))

# 6. Validasi kasus buruk
def bad(mutator, expect):
    d = raw.copy(); mutator(d)
    r = dl.validate_dataset(d)
    check(f"ditolak: {expect}", (not r.ok) and any(expect in e for e in r.errors), "; ".join(r.errors)[:90])

bad(lambda d: d.drop(columns=["Contract"], inplace=True), "Kolom wajib tidak ditemukan")
bad(lambda d: d.__setitem__("Contract", d["Contract"].replace({"One year": "Annual"})), "Contract")
bad(lambda d: d.__setitem__("tenure", d["tenure"].where(d.index != 5, -3)), "tenure")
bad(lambda d: d.__setitem__("MonthlyCharges", d["MonthlyCharges"].astype(object).where(d.index != 5, "abc")), "MonthlyCharges")
bad(lambda d: d.__setitem__("TotalCharges", d["TotalCharges"].where(d.index != 5, " ")), "TotalCharges")
bad(lambda d: d.__setitem__("customerID", d["customerID"].where(d.index != 5, d.loc[0, "customerID"])), "duplikat")
bad(lambda d: d.__setitem__("SeniorCitizen", d["SeniorCitizen"].where(d.index != 5, 7)), "SeniorCitizen")
bad(lambda d: d.loc[d.index == 0, "OnlineSecurity"].__class__ and d.__setitem__(
    "OnlineSecurity", d["OnlineSecurity"].where(d.index != 0, "No internet service")), "logika layanan")
check("ditolak: DataFrame kosong", not dl.validate_dataset(raw.iloc[0:0]).ok)

# 7. Mode unggahan: skor memakai bundle, kecocokan dengan notebook
tbl, label, w = dl.get_active_dataset("upload", raw)
m = scored.set_index("customerID").loc[tbl["customerID"]]
check("upload: kecocokan segmen 100%", (tbl["segment_id"].values == m["segment_id"].values).all())
check("upload: kolom analitik lengkap", {"churn_prob_cal", "risk_tier", "priority", "revenue_at_risk"} <= set(tbl.columns))
diff = np.abs(tbl["churn_prob_raw"].values - m["churn_prob_raw"].values).max()
print(f"INFO selisih maks prob mentah bundle(in-sample) vs OOF = {diff:.3f}  (diharapkan > 0)")

# 8. Upload tanpa Churn & tanpa customerID
r2 = dl.validate_dataset(raw.drop(columns=["Churn", "customerID"]).head(50))
check("upload tanpa Churn/ID: lolos + ID otomatis", r2.ok and r2.df["customerID"].iloc[0] == "UPLOAD-00001")
t2, _, _ = dl.get_active_dataset("upload", raw.drop(columns=["Churn"]).head(50))
check("upload tanpa Churn: Churn_flag = NaN", t2["Churn_flag"].isna().all())

# 9. Upload tidak kompatibel -> pesan ramah, bukan stack trace
try:
    dl.get_active_dataset("upload", raw.drop(columns=["tenure"]))
    check("upload tidak kompatibel memicu DataLoadError", False)
except dl.DataLoadError as e:
    check("upload tidak kompatibel -> DataLoadError ramah", "tenure" in str(e))

# 10. Cadangan URL jika file lokal hilang
orig = config.DATA_FILE
config.DATA_FILE = orig.with_name("tidak_ada.csv")
dl.load_raw_default.clear()
try:
    r = dl.load_raw_default()
    check("cadangan URL GitHub berfungsi", r.shape == (7043, 21))
except dl.DataLoadError as e:
    print("INFO cadangan URL tidak dapat diakses di lingkungan ini:", e)
finally:
    config.DATA_FILE = orig
    dl.load_raw_default.clear()

print(f"\nSemua {ok_count} pemeriksaan lulus.")
