"""Uji Stage 6 (Methodology & Model Performance):  python -m tests.test_stage6

Catatan: jalur interaktif diuji dengan DATA TIRUAN yang hanya hidup di dalam uji ini (bukan hasil analisis dan tidak dikirim
ke proyek). File prediksi asli datang dari sel Colab di tools/colab_export_test_predictions.py.
"""
import json, logging, sys, tempfile
from pathlib import Path
logging.disable(logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from streamlit.testing.v1 import AppTest
from utils import config, data_loader as dl, ui as _ui
from utils.analytics import data_quality
from utils.evaluation import metrics_at, roc_pr_summary
from sklearn.metrics import precision_score, recall_score, roc_auc_score

n_ok = 0
def check(name, cond, extra=""):
    global n_ok
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    assert cond, name
    n_ok += 1

def md(at): return " ".join(m.value for m in at.markdown)
raw = pd.read_csv(ROOT / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
ext = json.load(open(ROOT / "reports/metrics/notebook_extracts.json"))
summ = json.load(open(ROOT / "reports/metrics/metrics_summary.json"))

# ---- Konsistensi antar-artefak notebook
cm = ext["confusion_matrix_at_f2_threshold"]
kf2 = summ["test_metrics"]["@0.39 (F2)"]
check("confusion matrix: TP+FN+FP+TN = n_test", sum(cm.values()) == ext["split"]["n_test"])
check("recall dari confusion matrix = metrics_summary", abs(cm["TP"] / (cm["TP"] + cm["FN"]) - kf2["recall"]) < 1e-4)
check("precision dari confusion matrix = metrics_summary", abs(cm["TP"] / (cm["TP"] + cm["FP"]) - kf2["precision"]) < 1e-4)
check("churn di test = FN+TP = 374 (26.54% dari 1,409 ~ 374)", cm["TP"] + cm["FN"] == 374)
check("split: 5,634 latih + 1,409 uji = 7,043", ext["split"]["n_train"] + ext["split"]["n_test"] == 7043)
check("threshold F2 di extracts = metrics_summary", abs(ext["threshold_selection"]["threshold_f2"] - summ["threshold"]) < 0.005)

# ---- Kualitas data dihitung langsung
dq = data_quality(raw)
check("kualitas data: 11 TotalCharges kosong, semua tenure 0 dan churn No", dq["n_blank_tc"] == 11 and dq["blank_all_tenure0"] and dq["blank_all_no_churn"])
check("kualitas data: duplikat penuh 0, ID ganda 0, profil identik 22", (dq["n_full_dup"], dq["n_id_dup"], dq["n_profile_dup"]) == (0, 0, 22))
check("kualitas data: outlier IQR 0 dan logika 0 pelanggaran", sum(dq["iqr_outliers"].values()) == 0 and dq["n_logic_violations"] == 0)
check("kualitas data: korelasi TotalCharges 0.9996", abs(dq["corr_total_vs_expected"] - 0.9996) < 5e-4, f"{dq['corr_total_vs_expected']:.4f}")
check("kualitas data: churn 26.54%, rasio 2.77", abs(dq["churn_pct"] - 26.54) < 0.01 and abs(dq["class_ratio"] - 2.77) < 0.01)

# ---- Halaman (jalur statis: file prediksi asli sengaja disembunyikan sementara agar uji tidak bergantung padanya)
REAL = (dl.TEST_PRED_FILE, dl.TEST_META_FILE)
dl.TEST_PRED_FILE = Path(tempfile.mkdtemp()) / "tidak_ada.csv"
dl.load_test_predictions.clear()
at = AppTest.from_file(str(ROOT / "pages/7_Methodology.py"), default_timeout=180).run()
check("Methodology: tanpa exception", not at.exception, str([e.value for e in at.exception])[:200])
check("Methodology: 6 tab", len(at.tabs) == 6)
h = md(at)
check("Methodology: Silhouette 0.4029, DB 0.8765 tampil", "0.4029" in h and "0.8765" in h)
check("Methodology: alasan K-Means vs GMM dari angka", "syarat GMM tidak terpenuhi" in h and "0.3545" in h and "0.9983" in h)
check("Methodology: '6 dari 7' model dalam 1 SD", "6 dari 7" in h)
check("Methodology: versi statis + petunjuk aktivasi interaktif", "test_predictions.csv" in h and "TP = 324" in h)

# ---- Validasi file prediksi
def with_files(df, meta=None):
    tmp = Path(tempfile.mkdtemp())
    p, m = tmp / "test_predictions.csv", tmp / "test_predictions_meta.json"
    df.to_csv(p, index=False)
    if meta is not None: m.write_text(json.dumps(meta))
    dl.TEST_PRED_FILE, dl.TEST_META_FILE = p, m
    dl.load_test_predictions.clear()
    return dl.load_test_predictions()

orig = REAL
dl.TEST_PRED_FILE = Path(tempfile.mkdtemp()) / "tidak_ada.csv"; dl.load_test_predictions.clear()
check("prediksi test: file tidak ada -> None", dl.load_test_predictions() is None)

rng = np.random.RandomState(0)
ids = raw["customerID"].sample(1409, random_state=0).values
yt = np.zeros(1409, int); yt[:374] = 1; rng.shuffle(yt)
def synth(sep):
    return np.clip(0.35 + sep * (yt - 0.5) + rng.normal(0, 0.18, 1409), 0.001, 0.999)
good = pd.DataFrame({"customerID": ids, "y_true": yt, "XGBoost": synth(0.55), "LogReg (balanced)": synth(0.5)})
r = with_files(good, {"best_model": "XGBoost", "threshold": 0.39})
check("prediksi test: AUC tidak cocok dengan notebook -> ditolak", r["ok"] is False and any("roc_auc" in p for p in r["problems"]), "; ".join(r["problems"])[:80])
r = with_files(good.head(100), {"best_model": "XGBoost"})
check("prediksi test: jumlah baris salah -> ditolak", r["ok"] is False and any("Jumlah baris" in p for p in r["problems"]))
bad = good.copy(); bad.loc[0, "XGBoost"] = 1.7
check("prediksi test: probabilitas di luar 0-1 -> ditolak", with_files(bad, {"best_model": "XGBoost"})["ok"] is False)
bad = good.copy(); bad.loc[1, "customerID"] = bad.loc[0, "customerID"]
check("prediksi test: customerID duplikat -> ditolak", with_files(bad, {"best_model": "XGBoost"})["ok"] is False)
bad = good.copy(); bad.loc[0, "customerID"] = "TIDAK-ADA"
check("prediksi test: customerID tak dikenal -> ditolak", with_files(bad, {"best_model": "XGBoost"})["ok"] is False)
bad = good.copy(); bad.loc[0, "y_true"] = 3
check("prediksi test: y_true bukan 0/1 -> ditolak", with_files(bad, {"best_model": "XGBoost"})["ok"] is False)

# ---- Halaman dengan file tidak valid -> peringatan + jalur statis
with_files(good, {"best_model": "XGBoost"})
at = AppTest.from_file(str(ROOT / "pages/7_Methodology.py"), default_timeout=180).run()
check("Methodology: file prediksi tak valid -> peringatan, tanpa exception",
      not at.exception and any("tidak lolos validasi" in w.value for w in at.warning))

# ---- Jalur interaktif (DATA TIRUAN, validasi dilewati lewat patch)
fake = {"ok": True, "df": good, "meta": {"best_model": "XGBoost"}, "models": ["XGBoost", "LogReg (balanced)"], "problems": []}
orig_fn = dl.load_test_predictions
dl.load_test_predictions = lambda: fake
at = AppTest.from_file(str(ROOT / "pages/7_Methodology.py"), default_timeout=180).run()
check("Interaktif: tanpa exception", not at.exception, str([e.value for e in at.exception])[:200])
check("Interaktif: slider threshold dan pilihan model ada", len(at.slider) >= 1 and any(s.key == "m_cm_model" for s in at.selectbox))
check("Interaktif: tidak ada catatan versi statis", "akan aktif otomatis" not in " ".join(m.value for m in at.markdown))
at.slider(key="m_thr").set_value(0.60).run()
check("Interaktif: geser threshold tanpa exception", not at.exception)
m = metrics_at(good["y_true"].values, good["XGBoost"].values, 0.60)
sk_p = precision_score(good["y_true"], good["XGBoost"] >= 0.60); sk_r = recall_score(good["y_true"], good["XGBoost"] >= 0.60)
check("metrics_at sama dengan scikit-learn", abs(m["precision"] - sk_p) < 1e-12 and abs(m["recall"] - sk_r) < 1e-12)
check("roc_pr_summary sama dengan scikit-learn", abs(roc_pr_summary(good["y_true"], good["XGBoost"])["roc_auc"] - roc_auc_score(good["y_true"], good["XGBoost"])) < 1e-12)
at.multiselect(key="m_curve_models").set_value(["LogReg (balanced)"]).run()
check("Interaktif: pilih satu model kurva tanpa exception", not at.exception)
at.multiselect(key="m_curve_models").set_value([]).run()
check("Interaktif: tanpa model terpilih tidak crash", not at.exception)
dl.load_test_predictions = orig_fn
dl.TEST_PRED_FILE, dl.TEST_META_FILE = orig; dl.load_test_predictions.clear()

# ---- Upload mode tidak memengaruhi halaman metodologi
class Fake:
    def __init__(s, n, d): s.name, s._d = n, d
    def getvalue(s): return s._d
a = AppTest.from_file(str(ROOT / "pages/7_Methodology.py"), default_timeout=180)
a.session_state["data_mode"] = _ui.UPLOAD_LABEL
a.session_state["upload_file"] = Fake("x.csv", raw.drop(columns=["Churn"]).head(200).to_csv(index=False).encode())
a.run()
check("Methodology: mode unggahan tanpa exception", not a.exception, str([e.value for e in a.exception])[:200])
# ---- File prediksi ASLI dari Colab (jika sudah diletakkan di reports/metrics/)
dl.TEST_PRED_FILE, dl.TEST_META_FILE = REAL; dl.load_test_predictions.clear()
real = dl.load_test_predictions()
if real is None:
    print("INFO file test_predictions.csv asli belum ada di reports/metrics (mode statis dipakai). Lewati uji file asli.")
else:
    check("FILE ASLI: lolos validasi dan cocok dengan hasil notebook", real["ok"], "; ".join(real["problems"])[:150])
    if real["ok"]:
        y = real["df"]["y_true"].values; best = real["meta"].get("best_model") or summ["best_model"]
        m = metrics_at(y, real["df"][best].values, summ["threshold"])
        check("FILE ASLI: recall dan precision di threshold F2 sama dengan notebook",
              abs(m["recall"] - kf2["recall"]) < 5e-3 and abs(m["precision"] - kf2["precision"]) < 5e-3,
              f"recall {m['recall']:.4f} vs {kf2['recall']:.4f}; precision {m['precision']:.4f} vs {kf2['precision']:.4f}")
        check("FILE ASLI: confusion matrix sama dengan output notebook",
              (m["tp"], m["fn"], m["fp"], m["tn"]) == (cm["TP"], cm["FN"], cm["FP"], cm["TN"]), f"{(m['tp'], m['fn'], m['fp'], m['tn'])}")
        at = AppTest.from_file(str(ROOT / "pages/7_Methodology.py"), default_timeout=180).run()
        check("FILE ASLI: halaman memakai mode interaktif tanpa exception",
              not at.exception and "akan aktif otomatis" not in " ".join(x.value for x in at.markdown), str([e.value for e in at.exception])[:150])
print(f"\nSemua {n_ok} pemeriksaan lulus.")
