"""Uji Stage 3 dengan Streamlit AppTest: python -m tests.test_stage3"""
import logging, sys
from pathlib import Path
logging.disable(logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest

n_ok = 0
def check(name, cond, extra=""):
    global n_ok
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))
    assert cond, name
    n_ok += 1

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
check("app berjalan tanpa exception", not at.exception, str([e.value for e in at.exception]))
html = " ".join(m.value for m in at.markdown)
check("KPI total pelanggan 7,043", "7,043" in html)
check("KPI churn 1,869", "1,869" in html)
check("KPI churn rate 26.5%", "26.5%" in html)
check("KPI risiko tinggi 1,409", "1,409" in html)
check("rata-rata tenure 32.4", "32.4 bln" in html)
check("rata-rata tagihan $64.76", "$64.76" in html)

# Filter kontrak -> hanya Two year
ms = {w.key: w for w in at.multiselect}
ms["ov_con"].set_value(["Two year"]).run()
check("tanpa exception setelah filter", not at.exception)
html = " ".join(m.value for m in at.markdown)
check("filter Two year: total berubah menjadi 1,695", "1,695" in html)

# Filter kosong -> peringatan ramah, bukan crash
ms = {w.key: w for w in at.multiselect}
ms["ov_con"].set_value([]).run()
check("filter kosong: tanpa exception", not at.exception)
check("filter kosong: peringatan ditampilkan", any("Tidak ada pelanggan" in w.value for w in at.warning))
print(f"\nSemua {n_ok} pemeriksaan lulus.")

# ---- Mode unggahan (tanpa kolom Churn) dan unggahan tidak valid -------------------------------
import pandas as pd
from utils import ui as _ui

class FakeUpload:
    def __init__(self, name, data): self.name, self._d = name, data
    def getvalue(self): return self._d

raw = pd.read_csv(ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")

at2 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
at2.session_state["data_mode"] = _ui.UPLOAD_LABEL
at2.session_state["upload_file"] = FakeUpload("baru.csv", raw.drop(columns=["Churn"]).head(300).to_csv(index=False).encode())
at2.run()
check("upload tanpa Churn: tanpa exception", not at2.exception, str([e.value for e in at2.exception])[:200])
html2 = " ".join(m.value for m in at2.markdown)
check("upload: total 300 pelanggan", ">300<" in html2)
check("upload: KPI churn = n/a", ">n/a<" in html2)

at3 = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
at3.session_state["data_mode"] = _ui.UPLOAD_LABEL
at3.session_state["upload_file"] = FakeUpload("rusak.csv", raw.drop(columns=["tenure"]).to_csv(index=False).encode())
at3.run()
check("upload tidak valid: tanpa exception", not at3.exception)
check("upload tidak valid: pesan error ramah", any("tenure" in e.value for e in at3.error))
check("upload tidak valid: kembali ke dataset bawaan (7,043)", "7,043" in " ".join(m.value for m in at3.markdown))
print(f"\nTotal {n_ok} pemeriksaan lulus.")
