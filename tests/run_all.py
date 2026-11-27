"""Jalankan semua uji berurutan dan ringkas hasilnya.   python -m tests.run_all"""
import re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITES = [("test_stage2", "Loader data dan model"), ("test_stage3", "App, navigasi, Executive Overview"),
          ("test_stage4", "Segmentation, Explorer, Churn Intelligence"), ("test_stage5", "Simulator dan Retention Strategy"),
          ("test_stage6", "Methodology dan evaluasi"), ("test_stage7", "Portabilitas, ketahanan, properti prediksi")]
rows, failed = [], False
for mod, label in SUITES:
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", f"tests.{mod}"], cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    n_pass, n_fail = len(re.findall(r"^PASS ", out, re.M)), len(re.findall(r"^FAIL ", out, re.M))
    ok = p.returncode == 0 and n_fail == 0
    failed |= not ok
    rows.append((mod, label, n_pass, n_fail, "LULUS" if ok else "GAGAL", time.time() - t0))
    if not ok:
        print(f"\n--- Keluaran {mod} (gagal) ---"); print("\n".join(l for l in out.splitlines() if l.startswith(("FAIL", "Traceback", "AssertionError")) or "Error" in l)[:2500])
print(f"\n{'Suite':<13}{'Cakupan':<46}{'PASS':>5}{'FAIL':>6}  Status   Waktu")
for m, l, a, b, s, t in rows:
    print(f"{m:<13}{l:<46}{a:>5}{b:>6}  {s:<8} {t:>4.0f} dtk")
print(f"\nTotal: {sum(r[2] for r in rows)} pemeriksaan lulus, {sum(r[3] for r in rows)} gagal.")
sys.exit(1 if failed else 0)
