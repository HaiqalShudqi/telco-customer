# =====================================================================================================
# SEL TAMBAHAN COLAB: ekspor prediksi test set untuk halaman Methodology (grafik ROC/PR interaktif)
# Jalankan SETELAH Sel 46 (evaluasi test set) pada sesi Colab yang sama. Tidak melatih ulang apa pun.
# Hasil: test_predictions.csv dan test_predictions_meta.json (keduanya terunduh otomatis).
# =====================================================================================================
import json
import os

import numpy as np
import pandas as pd
from google.colab import files
from sklearn.metrics import average_precision_score, roc_auc_score

needed = ["p_test", "y_test", "X_test", "df_feat", "best_name", "THRESHOLD", "BETA", "test_df", "PATHS", "RS"]
missing = [n for n in needed if n not in globals()]
if missing:
    raise NameError(f"Variabel belum ada: {missing}. Jalankan ulang notebook dari Sel 2 sampai Sel 46 (sesi mungkin telah di-reset).")

# Satu baris per pelanggan di test set; satu kolom probabilitas per model (dari p_test pada Sel 46)
out = pd.DataFrame({"customerID": df_feat.loc[X_test.index, "customerID"].values,
                    "y_true": y_test.values.astype(int)})
for name, p in p_test.items():
    out[name] = np.asarray(p, dtype=float)

# Verifikasi: AUC dari file harus sama dengan hasil Sel 46
for name in p_test:
    auc = roc_auc_score(out["y_true"], out[name])
    ap = average_precision_score(out["y_true"], out[name])
    assert abs(auc - test_df.loc[name, "roc_auc"]) < 1e-4, f"ROC-AUC {name} tidak cocok"
    assert abs(ap - test_df.loc[name, "pr_auc"]) < 1e-4, f"PR-AUC {name} tidak cocok"
print(f"Verifikasi lulus untuk {len(p_test)} model. Baris: {len(out)}, churn: {int(out['y_true'].sum())}")

csv_path = os.path.join(PATHS["metrics"], "test_predictions.csv")
meta_path = os.path.join(PATHS["metrics"], "test_predictions_meta.json")
out.to_csv(csv_path, index=False, float_format="%.10g")
meta = {"best_model": best_name, "threshold": float(THRESHOLD), "beta": float(BETA), "n_test": int(len(out)),
        "n_test_churn": int(out["y_true"].sum()), "test_size": 0.2, "random_state": int(RS), "models": list(p_test.keys())}
with open(meta_path, "w") as fh:
    json.dump(meta, fh, indent=2)

files.download(csv_path)
files.download(meta_path)
print("Selesai. Letakkan kedua file di folder reports/metrics/ pada proyek Streamlit.")
