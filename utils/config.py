"""Konfigurasi terpusat: path relatif (aman untuk Streamlit Cloud) dan konstanta aplikasi."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Dataset bawaan di dalam repository (dibaca lewat path relatif, tanpa upload manual)
DATA_FILE = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
# Tabel pelanggan hasil notebook: segmen, probabilitas OOF, tier, prioritas
SCORED_FILE = ROOT / "data" / "processed" / "telco_customer_intelligence.csv"

MODELS_DIR = ROOT / "models"
BUNDLE_FILE = MODELS_DIR / "customer_intelligence_bundle.joblib"
# Cadangan portabel: model yang sama, booster XGBoost disimpan sebagai JSON teks (tidak sensitif platform)
PORTABLE_BUNDLE_FILE = MODELS_DIR / "customer_intelligence_bundle_portable.joblib"
XGB_JSON_FILE = MODELS_DIR / "xgb_booster.json"
# Jalur cadangan (model yang sama persis): booster XGBoost disimpan sebagai JSON teks agar portabel antar-OS
BUNDLE_PORTABLE_FILE = MODELS_DIR / "customer_intelligence_bundle_portable.joblib"
XGB_BOOSTER_FILE = MODELS_DIR / "xgb_booster.json"
SCHEMA_FILE = ROOT / "metadata" / "feature_schema.json"
DICT_FILE = ROOT / "metadata" / "data_dictionary.csv"
SESSION_FILE = ROOT / "metadata" / "session_info.json"
REPORTS = ROOT / "reports"

# Cadangan hanya jika file lokal hilang: salinan resmi IBM (isinya identik dengan data/WA_Fn-UseC_-Telco-Customer-Churn.csv)
FALLBACK_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")

APP_TITLE = "TELCO CUSTOMER INTELLIGENCE"
APP_SUBTITLE = "An Integrated Customer Analytics, Segmentation & Churn Prediction Platform"

# Versi saat model dilatih di Colab (pickle sensitif terhadap versi)
TRAINED_SKLEARN = "1.6.1"
TRAINED_XGBOOST = "3.4.1"

PRIORITY_ORDER = ["P1 - Selamatkan segera", "P2 - Retensi terstruktur", "P3 - Pantau", "P4 - Pertahankan"]
RISK_ORDER = ["Low", "Medium", "High"]
