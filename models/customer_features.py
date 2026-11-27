"""Utilitas fitur dan skoring Telco Customer Intelligence (dibuat otomatis oleh notebook)."""
import numpy as np
import pandas as pd

ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
TENURE_BINS = [-1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-12", "13-24", "25-48", "49-72"]
CONTRACT_ORD = {"Month-to-month": 0, "One year": 1, "Two year": 2}

NUM_FEATS = ["tenure", "MonthlyCharges", "TotalCharges", "n_addon_services", "total_service_count"]
CAT_FEATS = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
             "InternetService"] + ADDON_COLS + ["Contract", "PaperlessBilling", "PaymentMethod"]
MODEL_FEATS = NUM_FEATS + CAT_FEATS
CLUSTER_FEATS = ["tenure", "MonthlyCharges", "n_addon_services", "contract_ord"]


def build_features(raw):
    """Ubah data mentah (format CSV Telco) menjadi tabel fitur siap model. Tidak memakai kolom Churn."""
    d = raw.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"].replace(" ", np.nan), errors="coerce")
    d.loc[d["TotalCharges"].isna() & (d["tenure"] == 0), "TotalCharges"] = 0.0
    d["SeniorCitizen"] = d["SeniorCitizen"].replace({0: "No", 1: "Yes"})
    d["MultipleLines"] = d["MultipleLines"].replace({"No phone service": "No"})
    for c in ADDON_COLS:
        d[c] = d[c].replace({"No internet service": "No"})
    d["tenure_group"] = pd.cut(d["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(str)
    d["n_addon_services"] = (d[ADDON_COLS] == "Yes").sum(axis=1)
    d["total_service_count"] = ((d["PhoneService"] == "Yes").astype(int)
                                + (d["InternetService"] != "No").astype(int)
                                + d["n_addon_services"]
                                + (d["MultipleLines"] == "Yes").astype(int))
    d["avg_monthly_revenue"] = np.where(d["tenure"] > 0,
                                        d["TotalCharges"] / d["tenure"].replace(0, np.nan),
                                        d["MonthlyCharges"])
    d["annual_value_proxy"] = d["MonthlyCharges"] * 12
    d["contract_risk"] = (d["Contract"] == "Month-to-month").astype(int)
    d["contract_ord"] = d["Contract"].map(CONTRACT_ORD)
    return d


def assign_risk_tier(p, cut_med, cut_high):
    p = np.asarray(p)
    return np.where(p >= cut_high, "High", np.where(p >= cut_med, "Medium", "Low"))


def assign_value_tier(v, cut):
    return np.where(np.asarray(v) >= cut, "High", "Standard")


def assign_priority(risk, value):
    risk = np.asarray(risk); value = np.asarray(value)
    out = np.full(risk.shape, "P4 - Pertahankan", dtype=object)
    out[(risk == "Medium") & (value == "Standard")] = "P3 - Pantau"
    out[((risk == "High") & (value == "Standard")) | ((risk == "Medium") & (value == "High"))] = "P2 - Retensi terstruktur"
    out[(risk == "High") & (value == "High")] = "P1 - Selamatkan segera"
    return out


def score_customers(raw, bundle):
    """Skor pelanggan baru: segmen, probabilitas churn, tier risiko/nilai, prioritas. Tanpa training ulang."""
    f = build_features(raw)
    out = pd.DataFrame(index=f.index)
    if "customerID" in f.columns:
        out["customerID"] = f["customerID"]
    seg_raw = bundle["segment_pipe"].predict(f[CLUSTER_FEATS].astype(float))
    out["segment_id"] = pd.Series(seg_raw, index=f.index).map(bundle["segment_label_map"]).astype(int)
    out["segment_name"] = out["segment_id"].map(bundle["segment_names"])
    p_raw = bundle["churn_pipe"].predict_proba(f[MODEL_FEATS])[:, 1]
    out["churn_prob_raw"] = p_raw
    out["churn_prob_calibrated"] = bundle["calibrator"].predict(p_raw)
    out["risk_tier"] = assign_risk_tier(p_raw, bundle["risk_cut_med"], bundle["risk_cut_high"])
    out["value_tier"] = assign_value_tier(f["annual_value_proxy"], bundle["value_cut"])
    out["priority"] = assign_priority(out["risk_tier"], out["value_tier"])
    out["flag_churn_threshold"] = (p_raw >= bundle["threshold"]).astype(int)
    out["annual_value_proxy"] = f["annual_value_proxy"]
    out["revenue_at_risk"] = out["churn_prob_calibrated"] * f["annual_value_proxy"]
    return out
