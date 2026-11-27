"""Halaman 3: Customer Explorer (telusuri pelanggan individual)."""
import numpy as np
import pandas as pd
import streamlit as st

from utils import ui
from utils.analytics import ADDON_LABELS, ADDONS, has_churn
from utils.config import PRIORITY_ORDER, RISK_ORDER
from utils.model_loader import ModelLoadError, load_model_bundle
from utils.visualizations import seg_short

ui.bootstrap()
table, source_label, info = ui.get_dataset()
churn = has_churn(table)
ui.page_header("Customer Explorer", "Cari dan telusuri pelanggan individual: profil, segmen, probabilitas churn, dan prioritas.")
st.caption(f"Sumber data: {source_label}")

# ------------------------------------------------------------------ Filter
ui.section("Pencarian dan filter")
seg_names = table.sort_values("segment_id")["segment_name"].unique().tolist()
contracts = sorted(table["Contract"].unique())
internets = sorted(table["InternetService"].unique())
risks = [r for r in RISK_ORDER if r in table["risk_tier"].unique()]
t_min, t_max = int(table["tenure"].min()), int(table["tenure"].max())

r1a, r1b, r1c = st.columns([1.2, 2.2, 1.1])
query = r1a.text_input("Cari customerID", placeholder="mis. 7590-VHVEG", key="ex_q")
sel_seg = r1b.multiselect("Segmen", seg_names, default=seg_names, key="ex_seg")
status = r1c.selectbox("Status churn", ["Semua", "Churn", "Tidak churn"], key="ex_status", disabled=not churn,
                       help=None if churn else "Dataset tanpa kolom Churn.")
r2a, r2b, r2c = st.columns(3)
sel_con = r2a.multiselect("Kontrak", contracts, default=contracts, key="ex_con")
sel_int = r2b.multiselect("Layanan internet", internets, default=internets, key="ex_int")
sel_risk = r2c.multiselect("Tingkat risiko", risks, default=risks, key="ex_risk")
r3a, r3b = st.columns([2, 1.4])
sel_ten = r3a.slider("Tenure (bulan)", t_min, t_max, (t_min, t_max), key="ex_ten") if t_min < t_max else (t_min, t_max)
SORTS = {
    "Skor risiko: tertinggi lebih dulu": ("churn_prob_raw", False),
    "Skor risiko: terendah lebih dulu": ("churn_prob_raw", True),
    "Tagihan bulanan: tertinggi lebih dulu": ("MonthlyCharges", False),
    "Tenure: terlama lebih dulu": ("tenure", False),
    "customerID: A ke Z": ("customerID", True),
}
sort_label = r3b.selectbox("Urutkan", list(SORTS), key="ex_sort")

mask = (table["segment_name"].isin(sel_seg) & table["Contract"].isin(sel_con) & table["InternetService"].isin(sel_int)
        & table["risk_tier"].isin(sel_risk) & table["tenure"].between(*sel_ten))
if churn and status != "Semua":
    mask &= table["Churn_flag"] == (1 if status == "Churn" else 0)
if query.strip():
    mask &= table["customerID"].str.contains(query.strip(), case=False, regex=False)
col, asc = SORTS[sort_label]
df = table[mask].sort_values([col, "customerID"], ascending=[asc, True]).reset_index(drop=True)
ui.stop_if_empty(df)

# ------------------------------------------------------------------ Ringkasan hasil
s1, s2, s3 = st.columns(3)
s1.metric("Pelanggan cocok", f"{len(df):,}")
s2.metric("Churn aktual", f"{df['Churn_flag'].mean() * 100:.1f}%" if churn else "n/a")
s3.metric("Rata-rata probabilitas churn", f"{df['churn_prob_cal'].mean() * 100:.1f}%")
st.caption(f"{len(df):,} dari {len(table):,} pelanggan memenuhi filter.")

# ------------------------------------------------------------------ Tabel
ui.section("Daftar pelanggan", "Klik satu baris untuk melihat detail pelanggan di bawah tabel.")
view = pd.DataFrame({
    "customerID": df["customerID"],
    "Segmen": df["segment_name"].map(seg_short),
    "Churn": df["Churn_flag"].map({1: "Ya", 0: "Tidak"}).fillna("-") if churn else "-",
    "Tenure": df["tenure"],
    "Kontrak": df["Contract"],
    "Internet": df["InternetService"],
    "Tagihan/bln ($)": df["MonthlyCharges"],
    "Prob. churn (%)": df["churn_prob_cal"] * 100,
    "Skor model": df["churn_prob_raw"],
    "Risiko": df["risk_tier"],
    "Nilai": df["value_tier"],
    "Prioritas": df["priority"],
})
event = ui.dataframe(
    view, hide_index=True, on_select="rerun", selection_mode="single-row", key="ex_table", height=360,
    column_config={
        "Prob. churn (%)": st.column_config.ProgressColumn("Prob. churn (terkalibrasi)", format="%.1f%%", min_value=0, max_value=100),
        "Skor model": st.column_config.NumberColumn("Skor model (mentah)", format="%.3f"),
        "Tagihan/bln ($)": st.column_config.NumberColumn(format="$%.2f"),
    })
st.caption("Probabilitas terkalibrasi berasal dari skor mentah lewat kalibrasi isotonik (bertingkat, sehingga banyak nilai kembar). "
           "Tingkat risiko ditentukan dari skor mentah, dan pengurutan memakai skor mentah.")

# ------------------------------------------------------------------ Unduh
EXPORT_COLS = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
               "InternetService"] + ADDONS + ["Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
               "n_addon_services", "total_service_count", "annual_value_proxy", "segment_id", "segment_name", "churn_prob_raw",
               "churn_prob_cal", "risk_tier", "value_tier", "priority", "revenue_at_risk"] + (["Churn"] if churn else [])
export = df[[c for c in EXPORT_COLS if c in df.columns]]
st.download_button("Unduh data terfilter (CSV)", export.to_csv(index=False).encode("utf-8"),
                   file_name="telco_customers_filtered.csv", mime="text/csv", key="ex_dl")

# ------------------------------------------------------------------ Detail pelanggan
ui.section("Detail pelanggan")
picked = None
rows = (event.selection.rows if event is not None and hasattr(event, "selection") else []) or []
if rows and rows[0] < len(df):
    picked = df.iloc[rows[0]]["customerID"]
choice = st.selectbox("Atau pilih customerID dari hasil filter", df["customerID"].head(1000).tolist(), index=None,
                      placeholder="Ketik atau pilih ID", key="ex_pick")
if picked is None and choice:
    picked = choice
if picked is None and len(df) == 1:
    picked = df.iloc[0]["customerID"]
if picked is None:
    st.info("Pilih satu baris pada tabel, atau pilih customerID di atas, untuk melihat detail.")
    st.stop()

c = table[table["customerID"] == picked].iloc[0]
no_inet = c["InternetService"] == "No"
yn = lambda v: "Ya" if v == "Yes" else "Tidak"  # noqa: E731

with st.container(border=True):
    st.markdown(f"### {picked}")
    st.markdown(f"**Segmen:** {c['segment_name']}")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown("**Demografi**")
        st.markdown(f"- Gender: {c['gender']}\n- Senior citizen: {yn(c['SeniorCitizen'])}\n"
                    f"- Partner: {yn(c['Partner'])}\n- Tanggungan: {yn(c['Dependents'])}")
    with d2:
        st.markdown("**Kontrak**")
        st.markdown(f"- Jenis kontrak: {c['Contract']}\n- Tenure: {int(c['tenure'])} bulan ({c['tenure_group']})\n"
                    f"- Paperless billing: {yn(c['PaperlessBilling'])}\n- Pembayaran: {c['PaymentMethod']}")
    with d3:
        st.markdown("**Layanan**")
        addon_txt = "Tidak ada internet" if no_inet else ", ".join(ADDON_LABELS[a] for a in ADDONS if c[a] == "Yes") or "Tidak ada"
        st.markdown(f"- Telepon: {yn(c['PhoneService'])}; multiple lines: {yn(c['MultipleLines'])}\n"
                    f"- Internet: {c['InternetService']}\n- Layanan tambahan ({int(c['n_addon_services'])}): {addon_txt}")
    with d4:
        st.markdown("**Finansial**")
        st.markdown(f"- Tagihan bulanan: ${c['MonthlyCharges']:.2f}\n- Total tagihan: ${c['TotalCharges']:,.2f}\n"
                    f"- Proksi nilai tahunan: ${c['annual_value_proxy']:,.0f}\n- Revenue at risk (proksi): {c['revenue_at_risk']:,.1f}")

    st.divider()
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Probabilitas churn (terkalibrasi)", f"{c['churn_prob_cal'] * 100:.1f}%")
    p2.metric("Skor model (mentah)", f"{c['churn_prob_raw']:.3f}")
    p3.metric("Klasifikasi risiko", c["risk_tier"])
    p4.metric("Tingkat nilai", c["value_tier"])
    st.progress(float(min(max(c["churn_prob_cal"], 0), 1)), text=f"Prioritas retensi: {c['priority']}")
    if churn and not pd.isna(c["Churn_flag"]):
        st.caption(f"Status churn aktual dalam data: **{'Churn' if c['Churn_flag'] == 1 else 'Tidak churn'}**.")
    try:
        b = load_model_bundle()
        st.caption(f"Cutoff risiko (skor mentah): Medium ≥ {b['risk_cut_med']:.3f}, High ≥ {b['risk_cut_high']:.3f}. "
                   f"Nilai tinggi: proksi nilai tahunan ≥ ${b['value_cut']:,.0f}.")
    except ModelLoadError:
        pass
    if info["mode"] == "default":
        st.caption("Untuk dataset bawaan, skor adalah prediksi out-of-fold dari notebook (model tidak pernah melihat pelanggan ini saat memprediksinya).")
