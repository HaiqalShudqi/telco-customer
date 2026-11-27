"""Logika simulator prediksi churn: bentuk input mentah, validasi, skor memakai model tersimpan, dan skenario."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.data_loader import validate_dataset
from utils.model_loader import load_schema, score_raw

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


class InputError(Exception):
    """Input simulator tidak valid; `errors` berisi pesan yang aman ditampilkan."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def auto_total(tenure: int, monthly: float) -> float:
    """Perkiraan TotalCharges awal: tenure × tagihan bulanan (dapat diubah pengguna)."""
    return round(float(tenure) * float(monthly), 2)


def build_raw_row(inputs: dict) -> pd.DataFrame:
    """Satu baris data mentah dengan kolom persis seperti skema model."""
    cols = load_schema()["raw_input_columns"]
    row = {c: inputs.get(c) for c in cols}
    row["customerID"] = "SIMULASI"
    return pd.DataFrame([row])[["customerID"] + cols]


def validate_inputs(inputs: dict) -> tuple[pd.DataFrame, list[str]]:
    """Validasi dengan aturan yang sama seperti unggahan CSV. Mengembalikan (baris valid, peringatan)."""
    res = validate_dataset(build_raw_row(inputs))
    if not res.ok:
        raise InputError(res.errors)
    return res.df, res.warnings


def predict_one(inputs: dict) -> tuple[pd.Series, list[str], pd.DataFrame]:
    """Skor satu pelanggan memakai bundle tersimpan. Mengembalikan (hasil, peringatan, baris_valid)."""
    valid, warns = validate_inputs(inputs)
    scored = score_raw(valid)
    return scored.iloc[0], warns, valid


def scenario_table(inputs: dict) -> pd.DataFrame:
    """Skor ulang beberapa skenario perubahan satu faktor (tagihan dianggap tetap).

    Ini simulasi model berbasis asosiasi, bukan estimasi efek intervensi yang sebenarnya.
    """
    variants = [("Kondisi saat ini", {})]
    for c in ("Month-to-month", "One year", "Two year"):
        if c != inputs["Contract"]:
            variants.append((f"Kontrak diganti ke {c}", {"Contract": c}))
    if inputs["PaymentMethod"] == "Electronic check":
        for m in ("Bank transfer (automatic)", "Credit card (automatic)"):
            variants.append((f"Pembayaran diganti ke {m}", {"PaymentMethod": m}))
    if inputs["InternetService"] != "No":
        for a, lab in (("TechSupport", "Tech Support"), ("OnlineSecurity", "Online Security")):
            if inputs[a] == "No":
                variants.append((f"Menambah {lab}", {a: "Yes"}))
        if inputs["TechSupport"] == "No" and inputs["OnlineSecurity"] == "No":
            variants.append(("Menambah Tech Support dan Online Security", {"TechSupport": "Yes", "OnlineSecurity": "Yes"}))
    frames = []
    for name, change in variants:
        d = build_raw_row({**inputs, **change})
        d["customerID"] = name
        frames.append(d)
    batch = pd.concat(frames, ignore_index=True)
    res = validate_dataset(batch)
    if not res.ok:
        return pd.DataFrame()
    s = score_raw(res.df)
    out = pd.DataFrame({"Skenario": [v[0] for v in variants], "Probabilitas churn (%)": s["churn_prob_calibrated"].values * 100,
                        "Skor mentah": s["churn_prob_raw"].values, "Risiko": s["risk_tier"].values,
                        "Prioritas": s["priority"].values})
    out["Selisih vs saat ini (poin)"] = out["Probabilitas churn (%)"] - out["Probabilitas churn (%)"].iloc[0]
    return out


def score_percentile(raw_score: float, reference: pd.Series) -> float:
    """Persentase pelanggan referensi dengan skor mentah lebih rendah."""
    ref = np.asarray(reference, dtype=float)
    return float((ref < raw_score).mean() * 100)
