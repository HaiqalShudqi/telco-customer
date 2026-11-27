"""Uji Stage 5 (Churn Prediction Simulator dan Retention Strategy):  python -m tests.test_stage5"""
import logging, sys
from pathlib import Path
logging.disable(logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from streamlit.testing.v1 import AppTest
from utils import ui as _ui, simulator as sim
from utils.model_loader import load_model_bundle

n_ok = 0
def check(name, cond, extra=""):
    global n_ok
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    assert cond, name
    n_ok += 1

def run(page, **state):
    at = AppTest.from_file(str(ROOT / "pages" / page), default_timeout=120)
    for k, v in state.items(): at.session_state[k] = v
    return at.run()
def md(at): return " ".join(m.value for m in at.markdown)

raw = pd.read_csv(ROOT / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
scored = pd.read_csv(ROOT / "data/processed/telco_customer_intelligence.csv")
bundle = load_model_bundle(); cf = bundle["_cf"]

# =============================== Retention Strategy
at = run("6_Retention_Strategy.py")
check("Retention: tanpa exception", not at.exception, str([e.value for e in at.exception])[:200])
h = md(at)
check("Retention: risiko tinggi 1,409; nilai tinggi 1,771; P1 434", all(x in h for x in ["1,409", "1,771", ">434<"]))
check("Retention: revenue at risk total 1,667,798", "1,667,798" in h)
check("Retention: catatan proksi revenue at risk tampil", "indeks prioritisasi relatif" in h)

ref = pd.read_csv(ROOT / "reports/business/priority_summary.csv").set_index("priority")
dfs = [d.value for d in at.dataframe if "Revenue at risk (proksi)" in d.value.columns and "Definisi" in d.value.columns]
check("Retention: tabel ringkasan prioritas ada", len(dfs) == 1)
s = dfs[0]
check("Retention: jumlah per prioritas = priority_summary.csv", (s["Pelanggan"].values == ref.loc[s.index, "n"].values).all())
check("Retention: churn aktual per prioritas = notebook", np.allclose(s["Churn aktual (%)"].astype(float).values, ref.loc[s.index, "churn_rate_aktual"].round(1).values, atol=0.06))
check("Retention: porsi revenue at risk = notebook", np.allclose(s["Porsi revenue at risk (%)"].astype(float).values, ref.loc[s.index, "share_revenue_at_risk_%"].round(1).values, atol=0.06))
check("Retention: 3 tab rekomendasi segmen", len(at.tabs) == 3)
check("Retention: tombol unduh ada", len(at.get("download_button")) == 1)

at.multiselect(key="rs_prio").set_value(["P1 - Selamatkan segera"]).run()
check("Retention: filter P1 -> 434 pelanggan", ">434<" in md(at) and not at.exception)
at.multiselect(key="rs_prio").set_value([]).run()
check("Retention: filter kosong -> peringatan ramah", any("Tidak ada pelanggan" in w.value for w in at.warning) and not at.exception)

class Fake:
    def __init__(s, n, d): s.name, s._d = n, d
    def getvalue(s): return s._d
up = Fake("x.csv", raw.drop(columns=["Churn"]).head(300).to_csv(index=False).encode())
a = run("6_Retention_Strategy.py", data_mode=_ui.UPLOAD_LABEL, upload_file=up)
check("Retention: upload tanpa Churn tanpa exception", not a.exception, str([e.value for e in a.exception])[:200])
check("Retention: upload -> catatan rekomendasi hanya dataset bawaan", "hanya tersedia untuk dataset bawaan" in md(a))

# =============================== Simulator
at = run("5_Churn_Prediction.py")
check("Simulator: tanpa exception saat dibuka", not at.exception, str([e.value for e in at.exception])[:200])
check("Simulator: belum ada hasil sebelum tombol ditekan", "sim_result" not in at.session_state)

# Pelanggan 7590-VHVEG dimasukkan manual
c = raw[raw.customerID == "7590-VHVEG"].iloc[0]
vals = dict(sim_gender=c.gender, sim_senior=int(c.SeniorCitizen), sim_partner=c.Partner, sim_dep=c.Dependents, sim_tenure=int(c.tenure),
            sim_phone=c.PhoneService, sim_inet=c.InternetService, sim_OnlineSecurity=c.OnlineSecurity, sim_OnlineBackup=c.OnlineBackup,
            sim_DeviceProtection=c.DeviceProtection, sim_TechSupport=c.TechSupport, sim_StreamingTV=c.StreamingTV,
            sim_StreamingMovies=c.StreamingMovies, sim_contract=c.Contract, sim_paperless=c.PaperlessBilling,
            sim_pay=c.PaymentMethod, sim_monthly=float(c.MonthlyCharges))
for k, v in vals.items():
    w = at.selectbox(key=k) if k not in ("sim_tenure", "sim_monthly") else at.number_input(key=k)
    w.set_value(v)
at.run()
check("Simulator: telepon 'No' memaksa MultipleLines = tanpa layanan telepon", at.session_state["sim_ml"] == "No phone service")
check("Simulator: TotalCharges otomatis = tenure x tagihan", abs(at.session_state["sim_total"] - 1 * 29.85) < 1e-9)
at.button(key="sim_go").click().run()
check("Simulator: prediksi tanpa exception", not at.exception, str([e.value for e in at.exception])[:200])
got = at.session_state["sim_result"]["row"]
row = raw[raw.customerID == "7590-VHVEG"].drop(columns=["Churn"])
want = cf.score_customers(row, bundle).iloc[0]
same = all(got[k] == want[k] for k in ["churn_prob_raw", "churn_prob_calibrated", "segment_id", "risk_tier", "value_tier", "priority"])
check("Simulator: hasil sama persis dengan skor model tersimpan", same, f"p_raw={got['churn_prob_raw']:.4f}")
check("Simulator: hasil tampil (probabilitas, saran tindakan, skenario)", "Saran tindakan bisnis" in md(at) and any("skenario" in e.label for e in at.expander))

# Logika layanan: internet 'No' memaksa add-on
at.selectbox(key="sim_inet").set_value("No").run()
check("Simulator: internet 'No' memaksa semua add-on 'No internet service'",
      all(at.session_state[f"sim_{a}"] == "No internet service" for a in sim.ADDONS))
at.selectbox(key="sim_inet").set_value("DSL").run()
check("Simulator: internet kembali ada -> add-on kembali 'No'", all(at.session_state[f"sim_{a}"] == "No" for a in sim.ADDONS))
check("Simulator: input berubah -> catatan hasil usang", "Input telah diubah" in md(at))

# TotalCharges manual
at.checkbox(key="sim_auto_total").set_value(False).run()
at.number_input(key="sim_total").set_value(1234.5).run()
check("Simulator: TotalCharges manual dipertahankan", abs(at.session_state["sim_total"] - 1234.5) < 1e-9)
at.number_input(key="sim_tenure").set_value(100).run(); at.button(key="sim_go").click().run()
check("Simulator: tenure di luar rentang training -> peringatan, bukan error", any("di luar rentang" in w.value for w in at.warning) and not at.exception)

# Contoh cepat
at = run("5_Churn_Prediction.py")
at.button(key="preset_0").click().run()
check("Simulator: contoh cepat mengisi formulir", at.session_state["sim_tenure"] == 2 and at.session_state["sim_inet"] == "Fiber optic")
at.button(key="sim_go").click().run()
r = at.session_state["sim_result"]["row"]
check("Simulator: pelanggan baru fiber/bulanan berisiko tinggi", r["risk_tier"] == "High", f"{r['churn_prob_calibrated']*100:.1f}%")
at.button(key="preset_1").click().run(); at.button(key="sim_go").click().run()
r2 = at.session_state["sim_result"]["row"]
check("Simulator: pelanggan lama kontrak 2 tahun berisiko rendah", r2["risk_tier"] == "Low", f"{r2['churn_prob_calibrated']*100:.1f}%")
at.button(key="preset_2").click().run(); at.button(key="sim_go").click().run()
check("Simulator: contoh tanpa internet berjalan", not at.exception and at.session_state["sim_result"]["row"]["segment_id"] in (0, 1, 2))

# Validasi dan skenario (tanpa UI)
base = {"gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 5, "PhoneService": "No",
        "MultipleLines": "Yes", "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 50.0, "TotalCharges": 250.0}
try:
    sim.validate_inputs(base); check("validasi menolak telepon 'No' + MultipleLines 'Yes'", False)
except sim.InputError as e:
    check("validasi menolak telepon 'No' + MultipleLines 'Yes'", any("logika layanan" in m for m in e.errors))
ok = {**base, "PhoneService": "Yes"}
row_, w_, valid_ = sim.predict_one(ok)
sc = sim.scenario_table(ok)
check("skenario: baris pertama = kondisi saat ini = hasil utama", abs(sc.iloc[0]["Probabilitas churn (%)"] - row_["churn_prob_calibrated"] * 100) < 1e-9)
check("skenario: ada skenario kontrak dan pembayaran", sc["Skenario"].str.contains("Kontrak").any() and sc["Skenario"].str.contains("Pembayaran").any())
check("percentile skor wajar", 0 <= sim.score_percentile(0.5, scored["churn_prob_raw"]) <= 100)
print(f"\nSemua {n_ok} pemeriksaan lulus.")
