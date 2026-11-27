"""Pemuatan artefak model hasil Colab. Tidak ada training ulang di aplikasi ini."""
from __future__ import annotations

import importlib
import json
import sys
from types import ModuleType

import joblib
import pandas as pd
import streamlit as st

from utils import config


class ModelLoadError(Exception):
    """Error yang pesannya aman ditampilkan ke pengguna."""


def _import_features_module() -> ModuleType:
    """Import customer_features.py dari folder models/ (modul yang sama dengan saat training)."""
    if not (config.MODELS_DIR / "customer_features.py").exists():
        raise ModelLoadError("File models/customer_features.py tidak ditemukan di repository.")
    path = str(config.MODELS_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        return importlib.import_module("customer_features")
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError("Modul customer_features.py gagal dimuat.") from exc


def _check_versions() -> list[str]:
    """Bandingkan versi library runtime dengan versi training. Hanya peringatan, bukan error."""
    notes = []
    try:
        import sklearn
        if sklearn.__version__ != config.TRAINED_SKLEARN:
            notes.append(f"scikit-learn {sklearn.__version__} (model dilatih dengan {config.TRAINED_SKLEARN}).")
    except Exception:  # noqa: BLE001
        pass
    try:
        import xgboost
        if xgboost.__version__ != config.TRAINED_XGBOOST:
            notes.append(f"xgboost {xgboost.__version__} (model dilatih dengan {config.TRAINED_XGBOOST}).")
    except Exception:  # noqa: BLE001
        notes.append("Library xgboost tidak terpasang.")
    return notes


def _load_original():
    return joblib.load(config.BUNDLE_FILE)


def _load_portable():
    """Cadangan: bundle tanpa booster + booster XGBoost dari JSON. Model identik dengan bundle asli."""
    import xgboost as xgb
    bundle = joblib.load(config.BUNDLE_PORTABLE_FILE)
    booster = xgb.Booster()
    booster.load_model(bytearray(config.XGB_BOOSTER_FILE.read_bytes()))
    bundle["churn_pipe"].named_steps["clf"]._Booster = booster
    return bundle


def _load_portable() -> dict:
    """Muat bundle cadangan: pipeline tanpa booster + booster XGBoost dari JSON (model identik)."""
    import xgboost as xgb
    if not (config.PORTABLE_BUNDLE_FILE.exists() and config.XGB_JSON_FILE.exists()):
        raise FileNotFoundError("berkas cadangan portabel tidak ada")
    bundle = joblib.load(config.PORTABLE_BUNDLE_FILE)
    booster = xgb.Booster()
    booster.load_model(bytearray(config.XGB_JSON_FILE.read_bytes()))
    bundle["churn_pipe"].named_steps["clf"]._Booster = booster
    return bundle


@st.cache_resource(show_spinner="Memuat model...")
def load_model_bundle() -> dict:
    """Muat bundle (segmentasi, churn, kalibrator, cutoff, threshold) sekali per proses server.

    Urutan: bundle asli dari Colab -> jika gagal dibaca (mis. beda platform), bundle portabel
    yang berisi model yang sama persis.
    """
    if not config.BUNDLE_FILE.exists():
        raise ModelLoadError("File models/customer_intelligence_bundle.joblib tidak ditemukan.")
    cf = _import_features_module()
    source = "asli"
    try:
        bundle = joblib.load(config.BUNDLE_FILE)
    except Exception:  # noqa: BLE001
        try:
            bundle = _load_portable()
            source = "portabel"
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "Model gagal dimuat. Penyebab paling umum: versi scikit-learn/xgboost tidak sama "
                "dengan versi saat training, atau berkas model rusak. Periksa requirements.txt.") from exc
    required = {"segment_pipe", "segment_label_map", "segment_names", "churn_pipe", "calibrator",
                "risk_cut_med", "risk_cut_high", "value_cut", "threshold"}
    missing = required - set(bundle)
    if missing:
        raise ModelLoadError(f"Bundle tidak lengkap, komponen hilang: {sorted(missing)}")
    bundle["_version_notes"] = _check_versions()
    bundle["_cf"] = cf
    bundle["_source"] = source
    return bundle


@st.cache_resource
def load_schema() -> dict:
    if not config.SCHEMA_FILE.exists():
        raise ModelLoadError("File metadata/feature_schema.json tidak ditemukan.")
    with open(config.SCHEMA_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def score_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Skor data mentah (format CSV Telco) memakai model tersimpan. Tanpa training ulang."""
    bundle = load_model_bundle()
    try:
        return bundle["_cf"].score_customers(raw, bundle)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError("Prediksi gagal. Periksa bahwa data input sesuai skema model.") from exc
