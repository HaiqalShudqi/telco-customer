"""Pemuatan dataset (bawaan repo + unggahan opsional), validasi skema, dan tabel analitik."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from utils import config
from utils.model_loader import ModelLoadError, load_model_bundle, load_schema, score_raw

MAX_UPLOAD_ROWS = 20000


class DataLoadError(Exception):
    """Error yang pesannya aman ditampilkan ke pengguna."""


# ----------------------------------------------------------------------------------------------
# Dataset bawaan
# ----------------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat dataset...")
def load_raw_default() -> pd.DataFrame:
    """Dataset mentah bawaan. Urutan: file di repo (path relatif) -> URL GitHub (cadangan)."""
    if config.DATA_FILE.exists():
        try:
            return pd.read_csv(config.DATA_FILE)
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError("File dataset di repository tidak dapat dibaca sebagai CSV.") from exc
    try:
        return pd.read_csv(config.FALLBACK_URL)
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(
            "Dataset bawaan tidak ditemukan di folder data/ dan URL cadangan tidak dapat diakses.") from exc


@st.cache_data(show_spinner="Memuat hasil analisis...")
def load_scored_default() -> pd.DataFrame:
    """Tabel pelanggan hasil notebook (segmen, probabilitas out-of-fold, tier, prioritas).

    Skor OOF dipakai untuk dataset bawaan agar angka dashboard identik dengan laporan notebook.
    Model deployment di-refit pada seluruh data, sehingga skor in-sample akan sedikit berbeda.
    """
    if not config.SCORED_FILE.exists():
        raise DataLoadError("File data/processed/telco_customer_intelligence.csv tidak ditemukan.")
    df = pd.read_csv(config.SCORED_FILE)
    needed = {"customerID", "segment_id", "segment_name", "churn_prob_cal", "risk_tier",
              "value_tier", "priority", "revenue_at_risk", "Churn_flag"}
    if not needed.issubset(df.columns):
        raise DataLoadError(f"Kolom hasil analisis tidak lengkap: {sorted(needed - set(df.columns))}")
    raw = load_raw_default()
    if set(raw["customerID"]) != set(df["customerID"]):
        raise DataLoadError("customerID pada dataset bawaan tidak cocok dengan tabel hasil analisis.")
    return df


# ----------------------------------------------------------------------------------------------
# Validasi dataset unggahan
# ----------------------------------------------------------------------------------------------
@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    df: pd.DataFrame | None = None


ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def _examples(series: pd.Series, k: int = 4) -> str:
    return ", ".join(repr(v) for v in series.dropna().unique()[:k])


def validate_dataset(df: pd.DataFrame, schema: dict | None = None) -> ValidationResult:
    """Periksa dataset sebelum masuk ke model. Mengembalikan data bersih hanya jika tidak ada error."""
    schema = schema or load_schema()
    errors: list[str] = []
    warns: list[str] = []

    if df is None or df.empty:
        return ValidationResult(False, ["File kosong atau tidak memiliki baris data."])
    if len(df) > MAX_UPLOAD_ROWS:
        return ValidationResult(False, [f"Jumlah baris ({len(df):,}) melebihi batas {MAX_UPLOAD_ROWS:,}."])

    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]
    required = schema["raw_input_columns"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        return ValidationResult(False, [f"Kolom wajib tidak ditemukan: {', '.join(missing)}."])
    extra = [c for c in d.columns if c not in required + ["customerID", "Churn"]]
    if extra:
        warns.append(f"Kolom tambahan diabaikan: {', '.join(extra[:8])}.")

    # customerID
    if "customerID" not in d.columns:
        d["customerID"] = [f"UPLOAD-{i + 1:05d}" for i in range(len(d))]
        warns.append("Kolom customerID tidak ada; ID sementara dibuat otomatis.")
    d["customerID"] = d["customerID"].astype(str).str.strip()
    if d["customerID"].duplicated().any():
        errors.append(f"customerID duplikat ditemukan ({int(d['customerID'].duplicated().sum())} baris).")

    # Kategorikal
    for col, allowed in schema["categorical_allowed_values"].items():
        d[col] = d[col].astype("string").str.strip()
        bad = d[col].isna() | ~d[col].isin(allowed)
        if bad.any():
            errors.append(f"Kolom {col}: {int(bad.sum())} nilai tidak valid (contoh: {_examples(d.loc[bad, col])}). "
                          f"Nilai yang diizinkan: {', '.join(allowed)}.")
        d[col] = d[col].astype(object)

    # SeniorCitizen (0/1 atau Yes/No)
    sc = d["SeniorCitizen"].astype("string").str.strip().str.lower().map(
        {"0": 0, "1": 1, "0.0": 0, "1.0": 1, "no": 0, "yes": 1})
    if sc.isna().any():
        errors.append(f"Kolom SeniorCitizen: {int(sc.isna().sum())} nilai tidak valid (gunakan 0/1 atau Yes/No).")
    d["SeniorCitizen"] = sc

    # Numerik
    ten = pd.to_numeric(d["tenure"], errors="coerce")
    if ten.isna().any() or (ten < 0).any() or (ten != ten.round()).any():
        errors.append("Kolom tenure harus bilangan bulat tidak negatif (bulan).")
    mc = pd.to_numeric(d["MonthlyCharges"], errors="coerce")
    if mc.isna().any() or (mc < 0).any():
        errors.append("Kolom MonthlyCharges harus angka tidak negatif.")

    tc_raw = d["TotalCharges"].astype("string").str.strip().replace("", pd.NA)
    tc = pd.to_numeric(tc_raw, errors="coerce")
    blank_ok = tc.isna() & (ten == 0)            # pola dataset asli: belum pernah ditagih
    tc = tc.mask(blank_ok, 0.0)
    if tc.isna().any() or (tc < 0).any():
        errors.append(f"Kolom TotalCharges: {int((tc.isna() | (tc < 0)).sum())} nilai kosong/tidak valid "
                      "(nilai kosong hanya diterima jika tenure = 0).")
    if blank_ok.any():
        warns.append(f"{int(blank_ok.sum())} TotalCharges kosong dengan tenure = 0 diisi 0 (sesuai analisis).")

    # Konsistensi logika bisnis (aturan yang sama dengan pemeriksaan di notebook)
    if not errors:
        inet_no = d["InternetService"] == "No"
        bad_a = inet_no & (d[ADDONS] != "No internet service").any(axis=1)
        bad_b = ~inet_no & (d[ADDONS] == "No internet service").any(axis=1)
        bad_c = (d["PhoneService"] == "No") & (d["MultipleLines"] != "No phone service")
        bad_d = (d["PhoneService"] == "Yes") & (d["MultipleLines"] == "No phone service")
        n_bad = int((bad_a | bad_b | bad_c | bad_d).sum())
        if n_bad:
            errors.append(f"{n_bad} baris melanggar logika layanan (mis. tanpa internet tetapi punya add-on, "
                          "atau tanpa telepon tetapi MultipleLines bukan 'No phone service').")

    # Peringatan rentang di luar data training (bukan error)
    if not errors:
        for col, series in (("tenure", ten), ("MonthlyCharges", mc), ("TotalCharges", tc)):
            lo, hi = schema["numeric_ranges"][col]
            n_out = int(((series < lo) | (series > hi)).sum())
            if n_out:
                warns.append(f"{n_out} nilai {col} di luar rentang data training ({lo:g} sampai {hi:g}); "
                             "prediksi untuk baris tersebut kurang andal.")

    if errors:
        return ValidationResult(False, errors, warns)

    d["tenure"] = ten.astype(int)
    d["MonthlyCharges"] = mc.astype(float)
    d["TotalCharges"] = tc.astype(float)
    d["SeniorCitizen"] = d["SeniorCitizen"].astype(int)
    keep = ["customerID"] + required + (["Churn"] if "Churn" in d.columns else [])
    if "Churn" in d.columns:
        d["Churn"] = d["Churn"].astype("string").str.strip()
        if not d["Churn"].isin(["Yes", "No"]).all():
            warns.append("Kolom Churn diabaikan karena berisi nilai selain Yes/No.")
            keep.remove("Churn")
    return ValidationResult(True, [], warns, d[keep].reset_index(drop=True))


# ----------------------------------------------------------------------------------------------
# Tabel analitik (skema kolom sama dengan tabel hasil notebook)
# ----------------------------------------------------------------------------------------------
def build_analytic_table(raw_clean: pd.DataFrame) -> pd.DataFrame:
    """Skor data yang sudah tervalidasi dengan bundle dan susun tabel analitik seperti hasil notebook."""
    bundle = load_model_bundle()
    cf = bundle["_cf"]
    feats = cf.build_features(raw_clean)
    scored = score_raw(raw_clean)
    out = feats.copy()
    out["segment_id"] = scored["segment_id"].values
    out["segment_name"] = scored["segment_name"].values
    out["churn_prob_raw"] = scored["churn_prob_raw"].values
    out["churn_prob_cal"] = scored["churn_prob_calibrated"].values
    out["risk_tier"] = scored["risk_tier"].values
    out["value_tier"] = scored["value_tier"].values
    out["priority"] = scored["priority"].values
    out["revenue_at_risk"] = scored["revenue_at_risk"].values
    if "Churn" in out.columns:
        out["Churn_flag"] = (out["Churn"] == "Yes").astype(int)
    else:
        out["Churn"] = np.nan
        out["Churn_flag"] = np.nan
    return out


def get_active_dataset(mode: str = "default", uploaded_df: pd.DataFrame | None = None):
    """Satu pintu akses data untuk semua halaman.

    Mengembalikan (tabel_analitik, label_sumber, daftar_peringatan).
    Mode 'default'  : dataset repo + skor OOF dari notebook.
    Mode 'upload'   : CSV pengguna, divalidasi lalu diskor dengan model tersimpan.
    """
    if mode == "upload" and uploaded_df is not None:
        res = validate_dataset(uploaded_df)
        if not res.ok:
            raise DataLoadError("Dataset unggahan tidak kompatibel:\n- " + "\n- ".join(res.errors))
        try:
            table = build_analytic_table(res.df)
        except ModelLoadError as exc:
            raise DataLoadError(str(exc)) from exc
        return table, f"Dataset unggahan ({len(table):,} pelanggan)", res.warnings
    table = load_scored_default()
    return table, f"Dataset bawaan IBM Telco ({len(table):,} pelanggan)", []


@st.cache_data(show_spinner="Memvalidasi dan menskor dataset unggahan...", max_entries=3)
def process_upload(file_bytes: bytes) -> dict:
    """Baca, validasi, dan skor CSV unggahan. Di-cache per isi file; tidak pernah melempar error ke UI.

    Mengembalikan dict: ok, table, warnings, errors.
    """
    import io
    try:
        raw = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:  # noqa: BLE001
        return {"ok": False, "table": None, "warnings": [],
                "errors": ["File tidak dapat dibaca sebagai CSV. Pastikan formatnya CSV dengan header kolom."]}
    try:
        res = validate_dataset(raw)
    except ModelLoadError as exc:       # mis. feature_schema.json hilang
        return {"ok": False, "table": None, "warnings": [], "errors": [str(exc)]}
    if not res.ok:
        return {"ok": False, "table": None, "warnings": res.warnings, "errors": res.errors}
    try:
        table = build_analytic_table(res.df)
    except ModelLoadError as exc:
        return {"ok": False, "table": None, "warnings": res.warnings, "errors": [str(exc)]}
    return {"ok": True, "table": table, "warnings": res.warnings, "errors": []}


# ----------------------------------------------------------------------------------------------
# Hasil analisis dari notebook (file di folder reports/)
# ----------------------------------------------------------------------------------------------
@st.cache_data
def load_report_csv(relpath: str, index_col=None) -> pd.DataFrame | None:
    """Baca CSV hasil notebook. Mengembalikan None jika file tidak ada (halaman menampilkan pesan ramah)."""
    path = config.REPORTS / relpath
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=index_col)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data
def load_report_json(relpath: str) -> dict | None:
    import json
    path = config.REPORTS / relpath
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------------------------
# Prediksi test set (opsional; diekspor dari notebook dengan sel Colab di tools/)
# ----------------------------------------------------------------------------------------------
TEST_PRED_FILE = config.REPORTS / "metrics" / "test_predictions.csv"
TEST_META_FILE = config.REPORTS / "metrics" / "test_predictions_meta.json"


@st.cache_data
def load_test_predictions() -> dict | None:
    """Muat dan validasi prediksi test set. None jika file tidak ada.

    Mengembalikan dict: ok, df, meta, problems, models. Jika ok=False, halaman memakai gambar PNG sebagai cadangan.
    """
    import json
    if not TEST_PRED_FILE.exists():
        return None
    problems: list[str] = []
    try:
        df = pd.read_csv(TEST_PRED_FILE)
    except Exception:  # noqa: BLE001
        return {"ok": False, "df": None, "meta": {}, "models": [], "problems": ["File test_predictions.csv tidak dapat dibaca."]}
    meta = {}
    if TEST_META_FILE.exists():
        try:
            meta = json.loads(TEST_META_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            problems.append("test_predictions_meta.json tidak dapat dibaca.")
    models = [c for c in df.columns if c not in ("customerID", "y_true")]
    ext = load_report_json("metrics/notebook_extracts.json") or {}
    summ = load_report_json("metrics/metrics_summary.json") or {}

    if "y_true" not in df.columns or not models:
        problems.append("Kolom y_true atau kolom probabilitas model tidak ditemukan.")
    else:
        if not set(df["y_true"].unique()) <= {0, 1}:
            problems.append("y_true harus berisi 0/1.")
        if df[models].isna().any().any() or ((df[models] < 0) | (df[models] > 1)).any().any():
            problems.append("Probabilitas harus berada pada rentang 0 sampai 1 tanpa nilai kosong.")
        n_exp = ext.get("split", {}).get("n_test")
        if n_exp and len(df) != n_exp:
            problems.append(f"Jumlah baris ({len(df)}) tidak sama dengan jumlah test set di notebook ({n_exp}).")
        if "customerID" in df.columns:
            if df["customerID"].duplicated().any():
                problems.append("customerID duplikat pada file prediksi.")
            try:
                raw_ids = set(load_raw_default()["customerID"])
                if not set(df["customerID"]) <= raw_ids:
                    problems.append("Ada customerID yang tidak terdapat pada dataset bawaan.")
            except DataLoadError:
                pass
        best = meta.get("best_model") or summ.get("best_model")
        if best in models and not problems:
            from utils.evaluation import roc_pr_summary
            s = roc_pr_summary(df["y_true"].values, df[best].values)
            ref = summ.get("test_metrics", {}).get("@0.5", {})
            for k in ("roc_auc", "pr_auc"):
                if k in ref and abs(s[k] - ref[k]) > 1e-3:
                    problems.append(f"{k} dari file ({s[k]:.4f}) berbeda dari notebook ({ref[k]:.4f}); file tidak cocok dengan model terpilih.")
    return {"ok": not problems, "df": df, "meta": meta, "models": models, "problems": problems}
