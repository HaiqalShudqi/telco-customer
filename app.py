"""TELCO CUSTOMER INTELLIGENCE: titik masuk aplikasi (navigasi, tema, sidebar, pemilihan sumber data).

Jalankan:  streamlit run app.py
"""
import streamlit as st

from utils import config, ui

st.set_page_config(
    page_title="Telco Customer Intelligence",
    page_icon=":material/cell_tower:",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

# (file, judul, ikon). Halaman hanya didaftarkan jika file-nya ada, sehingga pengembangan bertahap aman.
PAGE_DEFS = [
    ("pages/1_Executive_Overview.py", "Executive Overview", ":material/dashboard:"),
    ("pages/2_Customer_Segmentation.py", "Customer Segmentation", ":material/hub:"),
    ("pages/3_Customer_Explorer.py", "Customer Explorer", ":material/manage_search:"),
    ("pages/4_Churn_Intelligence.py", "Churn Intelligence", ":material/query_stats:"),
    ("pages/5_Churn_Prediction.py", "Churn Prediction", ":material/online_prediction:"),
    ("pages/6_Retention_Strategy.py", "Retention Strategy", ":material/shield:"),
    ("pages/7_Methodology.py", "Methodology & Model", ":material/menu_book:"),
]
available = [d for d in PAGE_DEFS if (config.ROOT / d[0]).exists()]
pages = [st.Page(path, title=title, icon=icon, default=(i == 0)) for i, (path, title, icon) in enumerate(available)]

ui.render_sidebar()          # pemilihan sumber data + status model; hasilnya dibaca semua halaman
st.navigation(pages).run()
