"""Uji Stage 7: portabilitas, ketahanan, prediksi, dan kinerja.   python -m tests.test_stage7"""
import ast, contextlib, importlib, json, logging, os, re, sys, tempfile, time
from pathlib import Path
logging.disable(logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
from utils import config, data_loader as dl, model_loader as ml, simulator as sim, ui as _ui

n_ok = 0
def check(name, cond, extra=""):
    global n_ok
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    assert cond, name
    n_ok += 1

def clear_caches():
    st.cache_data.clear(); st.cache_resource.clear()

@contextlib.contextmanager
def patched(**attrs):
    """Ubah konstanta config sementara (mis. pura-pura file hilang), lalu pulihkan."""
    old = {k: getattr(config, k) for k in attrs}
    for k, v in attrs.items(): setattr(config, k, v)
    clear_caches()
    try: yield
    finally:
        for k, v in old.items(): setattr(config, k, v)
        clear_caches()

def tmp_missing(name="tidak_ada"): return Path(tempfile.mkdtemp()) / name
def run_page(page, **state):
    at = AppTest.from_file(str(ROOT / "pages" / page), default_timeout=180)
    for k, v in state.items(): at.session_state[k] = v
    return at.run()
def all_text(at):
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info]
    return " ".join(parts)

raw = pd.read_csv(ROOT / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PAGES = sorted(p.name for p in (ROOT / "pages").glob("*.py"))

# =============================================================== 1. Portabilitas dan keamanan (statis)
print("--- 1. Portabilitas dan keamanan")
code_files = [ROOT / "app.py"] + sorted((ROOT / "pages").glob("*.py")) + sorted((ROOT / "utils").glob("*.py")) + [ROOT / "models/customer_features.py"]
bad_paths = [(f.name, p) for f in code_files for p in ("/content/", "C:\\", "D:\\", "/home/", "/mnt/", "/Users/") if p in f.read_text(encoding="utf-8")]
check("tidak ada path absolut/Colab di kode aplikasi", not bad_paths, str(bad_paths))
sec = re.compile(r"""(api[_-]?key|secret|passw(or)?d|token)\s*[:=]\s*['"][^'"]+['"]""", re.I)
leaks = [(f.name) for f in code_files + list((ROOT / ".streamlit").glob("*.toml")) if sec.search(f.read_text(encoding="utf-8"))]
check("tidak ada rahasia/kredensial di kode atau konfigurasi", not leaks, str(leaks))
check("tidak ada secrets.toml di repo", not (ROOT / ".streamlit/secrets.toml").exists())
gi = (ROOT / ".gitignore").read_text()
check(".gitignore memuat .venv dan __pycache__", ".venv" in gi and "__pycache__" in gi)
check("set_page_config hanya di app.py dan ui.bootstrap", not any("set_page_config" in (ROOT / "pages" / p).read_text(encoding="utf-8") for p in PAGES))
check("tidak memakai API experimental Streamlit", not any("st.experimental_" in f.read_text(encoding="utf-8") for f in code_files))

app_src = (ROOT / "app.py").read_text()
declared = re.findall(r'"(pages/[^"]+\.py)"', app_src)
check("semua halaman di app.py ada (huruf besar/kecil persis)", all((ROOT / d).exists() and d.split("/")[1] in os.listdir(ROOT / "pages") for d in declared),
      f"{len(declared)} halaman")
consts = [config.DATA_FILE, config.SCORED_FILE, config.BUNDLE_FILE, config.PORTABLE_BUNDLE_FILE, config.XGB_JSON_FILE, config.SCHEMA_FILE, config.DICT_FILE]
check("file data/model ada dan nama cocok persis (aman untuk Linux)", all(p.exists() and p.name in os.listdir(p.parent) for p in consts))
sizes = [(p.stat().st_size, p) for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts]
check("tidak ada file > 25 MB", max(s for s, _ in sizes) < 25 * 2**20, f"terbesar {max(s for s, _ in sizes) / 1e6:.2f} MB")
total = sum(s for s, _ in sizes) / 1e6
check("ukuran proyek (tanpa .venv) < 50 MB", total < 50, f"{total:.1f} MB")

stdlib = set(sys.stdlib_module_names); local = {"utils", "customer_features"}
mods = set()
for f in code_files:
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import): mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module: mods.add(node.module.split(".")[0])
third = sorted(mods - stdlib - local)
req = (ROOT / "requirements.txt").read_text().lower()
PKG = {"sklearn": "scikit-learn"}
check("library pihak ketiga di kode = yang ada di requirements.txt",
      all(PKG.get(m, m) in req for m in third) and set(third) <= {"streamlit", "pandas", "numpy", "plotly", "joblib", "sklearn", "xgboost", "scipy"},
      str(third))
sess = json.loads((ROOT / "metadata/session_info.json").read_text())["library_versions"]
check("requirements memakai versi scikit-learn dan xgboost yang sama dengan saat training",
      f"scikit-learn=={sess['scikit-learn']}" in req and f"=={sess['xgboost']}" in req, f"{sess['scikit-learn']} / {sess['xgboost']}")

# =============================================================== 2. Semua halaman: default dan unggahan
print("--- 2. Smoke test semua halaman")
class Fake:
    def __init__(s, n, d): s.name, s._d = n, d
    def getvalue(s): return s._d
up_churn = Fake("dengan_churn.csv", raw.head(500).to_csv(index=False).encode())
up_nochurn = Fake("tanpa_churn.csv", raw.drop(columns=["Churn"]).head(200).to_csv(index=False).encode())
timings = {}
for label, state in (("bawaan", {}), ("unggahan+Churn", {"data_mode": _ui.UPLOAD_LABEL, "upload_file": up_churn}),
                     ("unggahan tanpa Churn", {"data_mode": _ui.UPLOAD_LABEL, "upload_file": up_nochurn})):
    for p in PAGES:
        t0 = time.time(); at = run_page(p, **state); dt = time.time() - t0
        timings[(label, p)] = dt
        txt = all_text(at)
        check(f"{p} [{label}]: tanpa exception dan tanpa 'Traceback'", not at.exception and "Traceback" not in txt,
              f"{dt:.1f} dtk" + (" | " + str([e.value for e in at.exception])[:150] if at.exception else ""))
check("semua halaman dimuat < 60 dtk", max(timings.values()) < 60, f"terlama {max(timings.values()):.1f} dtk")

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
check("app.py (navigasi) berjalan tanpa exception", not at.exception)
try:
    at.switch_page("pages/6_Retention_Strategy.py").run()
    check("pindah halaman lewat navigasi tanpa exception", not at.exception, "Retention Strategy")
except Exception as e:  # noqa: BLE001
    print("INFO switch_page tidak didukung di versi Streamlit ini:", type(e).__name__)

# =============================================================== 3. Unggahan: file buruk
print("--- 3. Ketahanan unggahan CSV")
def up(b): return dl.process_upload(b)
def csv_of(d, **k): return d.to_csv(index=False, **k).encode()
good = raw.drop(columns=["Churn"]).head(50)
cases = {
    "file kosong (0 byte)": (b"", False),
    "biner acak (bukan CSV)": (os.urandom(2000), False),
    "hanya header, tanpa baris": (csv_of(good.head(0)), False),
    "pemisah titik-koma": (csv_of(good).replace(b",", b";"), False),
    "kolom wajib hilang": (csv_of(good.drop(columns=["Contract"])), False),
    "kategori salah (huruf kecil 'yes')": (csv_of(good.assign(Partner=good["Partner"].str.lower())), False),
    "tenure kosong": (csv_of(good.assign(tenure=[np.nan] + list(good["tenure"][1:]))), False),
    "MonthlyCharges negatif": (csv_of(good.assign(MonthlyCharges=-5.0)), False),
    "customerID ganda": (csv_of(pd.concat([good, good.head(1)])), False),
    "melebihi 20.000 baris": (csv_of(pd.concat([good] * 401)), False),
    "valid: BOM UTF-8": (csv_of(good, encoding="utf-8-sig"), True),
    "valid: akhir baris Windows (CRLF)": (csv_of(good).replace(b"\n", b"\r\n"), True),
    "valid: kolom tambahan diabaikan": (csv_of(good.assign(catatan="x")), True),
    "valid: SeniorCitizen 'Yes/No'": (csv_of(good.assign(SeniorCitizen=good["SeniorCitizen"].map({0: "No", 1: "Yes"}))), True),
    "valid: spasi di sekitar kategori": (csv_of(good.assign(Contract=" " + good["Contract"] + " ")), True),
    "valid: tanpa kolom customerID": (csv_of(good.drop(columns=["customerID"])), True),
}
for name, (blob, expect) in cases.items():
    try:
        r = up(blob)
        ok = r["ok"] is expect and (expect or (r["errors"] and all(isinstance(e, str) for e in r["errors"])))
        check(f"unggahan {'diterima' if expect else 'ditolak ramah'}: {name}", ok, (r["errors"][0][:70] if r["errors"] else "ok"))
    except Exception as e:  # noqa: BLE001
        check(f"unggahan: {name} tidak boleh melempar exception", False, f"{type(e).__name__}: {e}")
t0 = time.time(); r = up(csv_of(raw.drop(columns=["Churn"]))); dt = time.time() - t0
check("unggahan 7.043 baris diskor < 30 dtk", r["ok"] and dt < 30, f"{dt:.1f} dtk")

# =============================================================== 4. Artefak hilang atau rusak
print("--- 4. Artefak hilang atau rusak")
with patched(REPORTS=Path(tempfile.mkdtemp())):
    for p in ("2_Customer_Segmentation.py", "4_Churn_Intelligence.py", "5_Churn_Prediction.py", "6_Retention_Strategy.py", "7_Methodology.py"):
        at = run_page(p); check(f"folder reports kosong: {p} tanpa exception", not at.exception, str([e.value for e in at.exception])[:120])
    at = run_page("7_Methodology.py"); check("folder reports kosong: Methodology memberi peringatan ramah", any("tidak ditemukan" in w.value for w in at.warning))

with patched(BUNDLE_FILE=tmp_missing("b.joblib"), PORTABLE_BUNDLE_FILE=tmp_missing("p.joblib")):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    check("model hilang: app (data bawaan) tetap berjalan tanpa exception", not at.exception)
    check("model hilang: pesan error ramah di sidebar", any("customer_intelligence_bundle.joblib" in e.value for e in at.error))
    at = run_page("5_Churn_Prediction.py"); check("model hilang: simulator berhenti dengan pesan ramah", not at.exception and any("Simulator tidak dapat dipakai" in e.value for e in at.error))
    at = run_page("6_Retention_Strategy.py", data_mode=_ui.UPLOAD_LABEL, upload_file=up_churn)
    check("model hilang + unggahan: pesan ramah, kembali ke data bawaan", not at.exception and any("tidak dapat dipakai" in e.value for e in at.error))

bad_bundle = tmp_missing("rusak.joblib"); bad_bundle.write_bytes(os.urandom(5000))
with patched(BUNDLE_FILE=bad_bundle, PORTABLE_BUNDLE_FILE=tmp_missing("p.joblib")):
    at = run_page("5_Churn_Prediction.py"); check("bundle rusak: pesan ramah tanpa exception", not at.exception and any("gagal dimuat" in e.value for e in at.error))

with patched(SCHEMA_FILE=tmp_missing("schema.json")):
    at = run_page("6_Retention_Strategy.py", data_mode=_ui.UPLOAD_LABEL, upload_file=up_churn)
    check("skema hilang + unggahan: tanpa exception", not at.exception, str([e.value for e in at.exception])[:120])

with patched(SCORED_FILE=tmp_missing("scored.csv")):
    at = run_page("1_Executive_Overview.py"); check("tabel hasil notebook hilang: pesan error ramah", not at.exception and len(at.error) >= 1)

orig_url = config.FALLBACK_URL
with patched(DATA_FILE=tmp_missing("data.csv"), FALLBACK_URL="http://localhost:9/tidak-ada.csv"):
    at = run_page("1_Executive_Overview.py"); check("dataset hilang + URL cadangan mati: pesan ramah", not at.exception and any("Dataset" in e.value for e in at.error))
    at = run_page("7_Methodology.py"); check("dataset hilang: Methodology tidak crash", not at.exception)

# =============================================================== 5. Uji properti prediksi (acak, 4.000 pelanggan)
print("--- 5. Uji properti prediksi")
rng = np.random.RandomState(7); N = 4000
cat = json.loads((ROOT / "metadata/feature_schema.json").read_text())["categorical_allowed_values"]
r = pd.DataFrame({"customerID": [f"R{i}" for i in range(N)], "gender": rng.choice(cat["gender"], N),
                  "SeniorCitizen": rng.choice([0, 1], N), "Partner": rng.choice(["Yes", "No"], N), "Dependents": rng.choice(["Yes", "No"], N),
                  "tenure": rng.randint(0, 73, N), "PhoneService": rng.choice(["Yes", "No"], N), "InternetService": rng.choice(cat["InternetService"], N),
                  "Contract": rng.choice(cat["Contract"], N), "PaperlessBilling": rng.choice(["Yes", "No"], N),
                  "PaymentMethod": rng.choice(cat["PaymentMethod"], N), "MonthlyCharges": np.round(rng.uniform(18.25, 118.75, N), 2)})
r["MultipleLines"] = np.where(r["PhoneService"] == "No", "No phone service", rng.choice(["Yes", "No"], N))
for a in sim.ADDONS:
    r[a] = np.where(r["InternetService"] == "No", "No internet service", rng.choice(["Yes", "No"], N))
r["TotalCharges"] = np.round(r["tenure"] * r["MonthlyCharges"] * rng.uniform(0.9, 1.1, N), 2)

res = dl.validate_dataset(r)
check("4.000 pelanggan acak lolos validator", res.ok, str(res.errors)[:100])
bundle = ml.load_model_bundle()
t0 = time.time(); s = ml.score_raw(res.df); dt = time.time() - t0
check("4.000 pelanggan diskor < 20 dtk", dt < 20, f"{dt:.1f} dtk")
p = s["churn_prob_raw"].values; pc = s["churn_prob_calibrated"].values
check("probabilitas mentah dan terkalibrasi berada di rentang 0..1", bool(((p >= 0) & (p <= 1)).all() and ((pc >= 0) & (pc <= 1)).all()))
order = np.argsort(p, kind="stable")
check("kalibrasi isotonik monoton (skor naik, probabilitas tidak turun; toleransi derau float32 1e-6)", bool((np.diff(pc[order]) >= -1e-6).all()), f"selisih terkecil {np.diff(pc[order]).min():.1e}")
exp_risk = np.where(p >= bundle["risk_cut_high"], "High", np.where(p >= bundle["risk_cut_med"], "Medium", "Low"))
check("tingkat risiko sesuai cutoff di bundle", bool((s["risk_tier"].values == exp_risk).all()))
exp_val = np.where(s["annual_value_proxy"].values >= bundle["value_cut"], "High", "Standard")
check("tingkat nilai sesuai cutoff di bundle", bool((s["value_tier"].values == exp_val).all()))
def prio(rk, v):
    if rk == "High" and v == "High": return "P1 - Selamatkan segera"
    if (rk == "High" and v == "Standard") or (rk == "Medium" and v == "High"): return "P2 - Retensi terstruktur"
    if rk == "Medium" and v == "Standard": return "P3 - Pantau"
    return "P4 - Pertahankan"
check("prioritas P1-P4 sesuai aturan risiko x nilai", all(prio(a, b) == c for a, b, c in zip(s["risk_tier"], s["value_tier"], s["priority"])))
check("segmen hanya 0/1/2 dan namanya sama dengan bundle",
      set(s["segment_id"].unique()) <= {0, 1, 2} and all(bundle["segment_names"][i] == n for i, n in zip(s["segment_id"], s["segment_name"])))
check("flag threshold = (skor >= threshold)", bool(((p >= bundle["threshold"]).astype(int) == s["flag_churn_threshold"].values).all()))
s2 = ml.score_raw(res.df)
check("deterministik: dua kali skor identik", bool(np.array_equal(s["churn_prob_raw"].values, s2["churn_prob_raw"].values)))
m2m, two = p[(r["Contract"] == "Month-to-month").values].mean(), p[(r["Contract"] == "Two year").values].mean()
check("sanity asosiasi: skor rata-rata kontrak bulanan > dua tahun", m2m > two, f"{m2m:.3f} vs {two:.3f}")
sh, lo = p[(r["tenure"] <= 6).values].mean(), p[(r["tenure"] >= 60).values].mean()
check("sanity asosiasi: tenure pendek > tenure panjang", sh > lo, f"{sh:.3f} vs {lo:.3f}")

base = {"gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 18.25, "TotalCharges": 0.0}
edge_ok = True
for ch in ({}, {"tenure": 72, "TotalCharges": 72 * 18.25}, {"MonthlyCharges": 118.75, "TotalCharges": 0.0}, {"tenure": 120, "TotalCharges": 2000.0},
           {"MonthlyCharges": 0.0, "tenure": 5, "TotalCharges": 0.0}):
    try:
        row, w, _ = sim.predict_one({**base, **ch}); edge_ok &= 0 <= row["churn_prob_calibrated"] <= 1
    except Exception as e:  # noqa: BLE001
        edge_ok = False; print("  edge gagal:", ch, type(e).__name__, e)
check("nilai batas (tenure 0/72/120, tagihan 0/18.25/118.75) diprediksi tanpa error", edge_ok)

# =============================================================== 6. Ringkasan kinerja
print("--- 6. Ringkasan waktu muat halaman (detik, AppTest, sudah termasuk cache dingin pertama)")
tab = pd.Series({f"{k[1]} [{k[0]}]": round(v, 1) for k, v in timings.items()})
print(tab.sort_values(ascending=False).head(6).to_string())
print(f"\nSemua {n_ok} pemeriksaan lulus.")
