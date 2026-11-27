"""Ekstrak hasil yang hanya ada sebagai teks di output notebook (log-rank dan probabilitas bertahan Kaplan-Meier).

Dipakai sekali oleh pengembang; aplikasi hanya membaca JSON hasilnya.
Pemakaian:  python tools/extract_notebook_outputs.py path/ke/Telco_Customer_Intelligence.ipynb
"""
import json
import re
import sys
from pathlib import Path

nb_path = Path(sys.argv[1])
out_path = Path(__file__).resolve().parent.parent / "reports" / "metrics" / "survival_summary.json"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

text = ""
for c in nb["cells"]:
    if c["cell_type"] != "code" or "survdiff" not in "".join(c["source"]):
        continue
    for o in c.get("outputs", []):
        text += "".join(o.get("text", []))
        text += "\n" + "".join(o.get("data", {}).get("text/plain", []))

logrank = {}
for m in re.finditer(r"Log-rank\s+(\w+)\s*:\s*chi2 = ([\d,\.]+), p = ([\d\.e\+\-]+)", text):
    logrank[m.group(1)] = {"chi2": float(m.group(2).replace(",", "")), "p_value": float(m.group(3))}

surv = {}
for m in re.finditer(r"^(Month-to-month|One year|Two year)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s*$", text, re.M):
    surv[m.group(1)] = {"S(12)": float(m.group(2)), "S(24)": float(m.group(3)), "S(48)": float(m.group(4))}

if len(logrank) != 3 or len(surv) != 3:
    raise SystemExit(f"Ekstraksi tidak lengkap: logrank={logrank}, surv={surv}")
out_path.write_text(json.dumps({"logrank": logrank, "survival_by_contract": surv,
                                "sumber": "Output sel Kaplan-Meier pada notebook (Tahap 7)"}, indent=2), encoding="utf-8")
print("Tersimpan:", out_path)
print(json.dumps({"logrank": logrank, "survival_by_contract": surv}, indent=2))


# ------------------------------------------------------------------------------------------------
# Bagian 2: hasil lain yang hanya ada di teks output notebook -> reports/metrics/notebook_extracts.json
# ------------------------------------------------------------------------------------------------
def cell_text(marker: str) -> str:
    """Gabungan teks output dari sel kode yang outputnya mengandung `marker`."""
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        t = ""
        for o in c.get("outputs", []):
            t += "".join(o.get("text", [])) + "\n" + "".join(o.get("data", {}).get("text/plain", [])) + "\n"
        if marker in t:
            return t
    raise SystemExit(f"Output dengan penanda '{marker}' tidak ditemukan di notebook.")


num = lambda s: float(s.replace(",", ""))  # noqa: E731
ext = {"sumber": "Teks output sel notebook (Tahap 6 dan 8)"}

t = cell_text("Perbandingan linkage")
ext["linkage_comparison"] = [
    {"linkage": m.group(1), "cophenetic_corr": float(m.group(2)), "silhouette_at_k": float(m.group(3)),
     "min_cluster_share": float(m.group(4))}
    for m in re.finditer(r"^\d+\s+(ward|average|complete)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", t, re.M)]
ext["hierarchical_by_k"] = [
    {"k": int(m.group(1)), "silhouette": float(m.group(2)), "davies_bouldin": float(m.group(3)),
     "calinski_harabasz": num(m.group(4)), "min_cluster_share": float(m.group(5))}
    for m in re.finditer(r"^(\d)\s+([\d.]+)\s+([\d.]+)\s+([\d,]+\.\d+)\s+([\d.]+)\s*$", t, re.M)]

t = cell_text("BIC terendah")
first = {int(m.group(1)): m for m in re.finditer(
    r"^(\d+)\s+(full|diag)\s+(\d)\s+(-?[\d,]+\.\d+)\s+(-?[\d,]+\.\d+)\s+([\d.]+)\s+([\d.]+)", t, re.M)}
second = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"^(\d+)\s+([\d.]+)\s*$", t, re.M)}
ext["gmm"] = [
    {"cov": m.group(2), "k": int(m.group(3)), "AIC": num(m.group(4)), "BIC": num(m.group(5)), "silhouette": float(m.group(6)),
     "mean_max_posterior": float(m.group(7)), "min_cluster_share": second.get(i)} for i, m in sorted(first.items())]
m = re.search(r"BIC terendah: cov=(\w+), k=(\d+)", t)
ext["gmm_lowest_bic"] = {"cov": m.group(1), "k": int(m.group(2))}

t = cell_text("scale_pos_weight")
m = re.search(r"Train: \((\d+), \d+\) \(churn ([\d.]+)%\) \| Test: \((\d+), \d+\) \(churn ([\d.]+)%\)", t)
ext["split"] = {"n_train": int(m.group(1)), "churn_train_pct": float(m.group(2)), "n_test": int(m.group(3)),
                "churn_test_pct": float(m.group(4)), "test_size": 0.2, "random_state": 42}
ext["scale_pos_weight"] = float(re.search(r"scale_pos_weight = ([\d.]+)", t).group(1))

t = cell_text("THRESHOLD (F2-optimal)")
m = re.search(r"THRESHOLD \(F2-optimal\) = ([\d.]+) -> OOF precision ([\d.]+), recall ([\d.]+) \| threshold F1-optimal = ([\d.]+)", t)
ext["threshold_selection"] = {"threshold_f2": float(m.group(1)), "oof_precision": float(m.group(2)), "oof_recall": float(m.group(3)),
                              "threshold_f1": float(m.group(4)), "beta": 2.0}

t = cell_text("TP=")
m = re.search(r"TP=(\d+).*?FN=(\d+).*?FP=(\d+).*?TN=(\d+)", t)
ext["confusion_matrix_at_f2_threshold"] = {"TP": int(m.group(1)), "FN": int(m.group(2)), "FP": int(m.group(3)), "TN": int(m.group(4))}
ext["baseline_pr_auc_test"] = float(re.search(r"Baseline PR-AUC .*? = ([\d.]+)", t if "Baseline" in t else cell_text("Baseline PR-AUC")).group(1))

# Parameter eksperimen dari kode notebook (bukan output)
src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
ext["tuning"] = {"n_iter_randomized_search": int(re.search(r"N_ITER = (\d+)", src).group(1)),
                 "cv_folds": int(re.search(r"cv5 = StratifiedKFold\(n_splits=(\d+)", src).group(1)),
                 "scoring_for_selection": "average_precision (PR-AUC)"}

if not (len(ext["linkage_comparison"]) == 3 and len(ext["hierarchical_by_k"]) == 7 and len(ext["gmm"]) == 14):
    raise SystemExit("Ekstraksi bagian 2 tidak lengkap.")
out2 = out_path.parent / "notebook_extracts.json"
out2.write_text(json.dumps(ext, indent=2), encoding="utf-8")
print("Tersimpan:", out2)
