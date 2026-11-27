"""Uji Stage 4 (Segmentation, Explorer, Churn Intelligence):  python -m tests.test_stage4"""
import logging, sys
from pathlib import Path
logging.disable(logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from streamlit.testing.v1 import AppTest
from utils import ui as _ui
from utils.analytics import segment_profile, rate_table, gap_ranking, wilson_ci

n_ok = 0
def check(name, cond, extra=""):
    global n_ok
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    assert cond, name
    n_ok += 1

def run(page, **state):
    at = AppTest.from_file(str(ROOT / "pages" / page), default_timeout=120)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()

def md(at): return " ".join(m.value for m in at.markdown)
def metrics(at): return {m.label: m.value for m in at.metric}

scored = pd.read_csv(ROOT / "data/processed/telco_customer_intelligence.csv")
ref = pd.read_csv(ROOT / "reports/segmentation/segment_profiles.csv", index_col=0)

# ---- Profil hitung-ulang harus sama dengan segment_profiles.csv dari notebook
prof = segment_profile(scored)
common = [c for c in ref.columns if c in prof.columns and c != "segment_name"]
diff = (prof[common].astype(float) - ref[common].astype(float)).abs().max().max()
check("profil segmen = segment_profiles.csv notebook", diff < 1e-6, f"selisih maks {diff:.2e}")
check("nama segmen sama dengan notebook", (prof["segment_name"] == ref["segment_name"]).all())
lo, hi = wilson_ci(1869, 7043)
check("CI Wilson churn keseluruhan wajar", 25.5 < lo < 26.5 < hi < 27.6, f"{lo:.2f}-{hi:.2f}")

# ---- Halaman 2
at = run("2_Customer_Segmentation.py")
check("Segmentation: tanpa exception", not at.exception, str([e.value for e in at.exception])[:200])
h = md(at)
check("Segmentation: ukuran 3,651 / 1,165 / 2,227", all(x in h for x in ["3,651", "1,165", "2,227"]))
check("Segmentation: churn 42.4 / 2.0 / 13.4", all(x in h for x in ["42.4", "2.0", "13.4"]))
check("Segmentation: 3 tab interpretasi", len(at.tabs) == 3)
at.multiselect(key="sg_sel").set_value([at.multiselect(key="sg_sel").options[0]]).run()
check("Segmentation: filter 1 segmen tanpa exception", not at.exception)
at.multiselect(key="sg_sel").set_value([]).run()
check("Segmentation: filter kosong -> peringatan", any("minimal satu" in w.value for w in at.warning))

# ---- Halaman 3
at = run("3_Customer_Explorer.py")
check("Explorer: tanpa exception", not at.exception, str([e.value for e in at.exception])[:200])
check("Explorer: 7,043 pelanggan", metrics(at)["Pelanggan cocok"] == "7,043")
at.text_input(key="ex_q").set_value("7590-VHVEG").run()
check("Explorer: cari ID -> 1 pelanggan + detail", metrics(at)["Pelanggan cocok"] == "1" and "### 7590-VHVEG" in md(at))
at.text_input(key="ex_q").set_value("zzz-tidak-ada").run()
check("Explorer: ID tidak ada -> peringatan ramah", any("Tidak ada pelanggan" in w.value for w in at.warning) and not at.exception)
at.text_input(key="ex_q").set_value("").run()
at.multiselect(key="ex_risk").set_value(["High"]).run()
check("Explorer: risiko High = 1,409", metrics(at)["Pelanggan cocok"] == "1,409")
at.selectbox(key="ex_status").set_value("Churn").run()
exp = int(((scored.risk_tier == "High") & (scored.Churn_flag == 1)).sum())
check(f"Explorer: High + churn = {exp:,}", metrics(at)["Pelanggan cocok"] == f"{exp:,}")
check("Explorer: tombol unduh CSV ada", len(at.get("download_button")) == 1)
# Urutan: skor tertinggi lebih dulu
at = run("3_Customer_Explorer.py")
at.selectbox(key="ex_sort").set_value("Skor risiko: terendah lebih dulu").run()
check("Explorer: ganti urutan tanpa exception", not at.exception)

# ---- Halaman 4
at = run("4_Churn_Intelligence.py")
check("Churn: tanpa exception", not at.exception, str([e.value for e in at.exception])[:200])
h = md(at)
chi = pd.read_csv(ROOT / "reports/metrics/stat_chi_square.csv")
n_sig = int(chi["signifikan_holm"].sum())
check(f"Churn: '{n_sig} dari {len(chi)}' variabel signifikan", f"**{n_sig} dari {len(chi)}**" in h)
check("Churn: Cramér's V Contract 0.410 tampil", "Contract (V = 0.410)" in h)
check("Churn: ukuran efek Mann-Whitney tampil", "tenure:** median pelanggan churn 10.00 vs 38.00" in h)
check("Churn: survival 70% bulan-12 kontrak bulanan", "70%" in h)
rt = rate_table(scored.assign(), "Contract")
check("rate_table Contract: Month-to-month 42.7%", abs(rt.set_index("Contract").loc["Month-to-month", "churn_rate"] - 42.71) < 0.02)
check("gap_ranking: Contract memimpin", gap_ranking(scored).iloc[0]["variabel"] in ("Contract", "tenure_group"))

# ---- Upload tanpa Churn
class Fake:
    def __init__(s, n, d): s.name, s._d = n, d
    def getvalue(s): return s._d
raw = pd.read_csv(ROOT / "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
up = Fake("x.csv", raw.drop(columns=["Churn"]).head(300).to_csv(index=False).encode())
st = {"data_mode": _ui.UPLOAD_LABEL, "upload_file": up}
for page in ["2_Customer_Segmentation.py", "3_Customer_Explorer.py", "4_Churn_Intelligence.py"]:
    a = run(page, **st)
    check(f"upload tanpa Churn: {page} tanpa exception", not a.exception, str([e.value for e in a.exception])[:200])
a = run("4_Churn_Intelligence.py", **st)
check("upload tanpa Churn: tab churn menampilkan catatan", "tidak memiliki kolom Churn" in md(a))
print(f"\nSemua {n_ok} pemeriksaan lulus.")
