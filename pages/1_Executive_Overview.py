"""Halaman 1: Executive Overview."""
import streamlit as st

from utils import ui
from utils.analytics import compute_kpis, executive_summary, has_churn
from utils.config import RISK_ORDER
from utils.model_loader import ModelLoadError, load_model_bundle
from utils.visualizations import (CHURN_COLORS, RISK_COLORS, donut, histogram, seg_short, segment_color_map,
                                  stacked_count_bar, tier_bar)

ui.bootstrap()
table, source_label, info = ui.get_dataset()
ui.page_header("Executive Overview", "Gambaran menyeluruh basis pelanggan: ukuran, churn, nilai tagihan, segmen, dan risiko.")
st.caption(f"Sumber data: {source_label}")

# ------------------------------------------------------------------ Filter
ui.section("Filter", "Semua angka dan grafik di halaman ini mengikuti filter berikut.")
seg_names = table.sort_values("segment_id")["segment_name"].unique().tolist()
contracts = sorted(table["Contract"].unique())
internets = sorted(table["InternetService"].unique())
risks = [r for r in RISK_ORDER if r in table["risk_tier"].unique()]
t_min, t_max = int(table["tenure"].min()), int(table["tenure"].max())

f1, f2, f3 = st.columns([2.2, 1.4, 1.2])
sel_seg = f1.multiselect("Segmen", seg_names, default=seg_names, key="ov_seg")
sel_con = f2.multiselect("Kontrak", contracts, default=contracts, key="ov_con")
sel_int = f3.multiselect("Layanan internet", internets, default=internets, key="ov_int")
f4, f5 = st.columns([1.4, 2.2])
sel_risk = f4.multiselect("Tingkat risiko", risks, default=risks, key="ov_risk")
sel_ten = f5.slider("Tenure (bulan)", t_min, t_max, (t_min, t_max), key="ov_ten") if t_min < t_max else (t_min, t_max)

df = table[
    table["segment_name"].isin(sel_seg) & table["Contract"].isin(sel_con) & table["InternetService"].isin(sel_int)
    & table["risk_tier"].isin(sel_risk) & table["tenure"].between(*sel_ten)
]
ui.stop_if_empty(df)

# ------------------------------------------------------------------ KPI
k, k_all = compute_kpis(df), compute_kpis(table)
filtered = len(df) != len(table)
ui.kpi_row([
    ("Total pelanggan", f"{k['n']:,}", f"dari {k_all['n']:,} pelanggan" if filtered else "seluruh basis", "#0F6E8C"),
    ("Pelanggan churn", f"{k['n_churn']:,}" if k["has_churn"] else "n/a",
     "" if k["has_churn"] else "data tanpa kolom Churn", "#D1495B"),
    ("Churn rate", f"{k['churn_rate']:.1f}%" if k["has_churn"] else "n/a",
     f"seluruh basis: {k_all['churn_rate']:.1f}%" if k["has_churn"] and filtered else "", "#D1495B"),
    ("Rata-rata tenure", f"{k['tenure_mean']:.1f} bln", f"seluruh basis: {k_all['tenure_mean']:.1f}" if filtered else "", "#6C5B9E"),
    ("Rata-rata tagihan bulanan", f"${k['monthly_mean']:.2f}",
     f"seluruh basis: ${k_all['monthly_mean']:.2f}" if filtered else "", "#F2A541"),
    ("Pelanggan risiko tinggi", f"{k['n_high_risk']:,}", f"{k['pct_high_risk']:.1f}% dari yang ditampilkan", "#14213D"),
])

# ------------------------------------------------------------------ Grafik baris 1
ui.section("Komposisi pelanggan")
seg_df = df.assign(segmen=df["segment_name"].map(seg_short))
seg_cmap = segment_color_map(table)
c1, c2, c3 = st.columns(3)
with c1:
    ui.plot(donut(seg_df, "segmen", seg_cmap, title="Distribusi segmen",
                  order=sorted(seg_df["segmen"].unique()), hole_text=f"{len(df):,}<br>pelanggan"), key="ov_seg_donut")
with c2:
    if has_churn(df):
        ch = df.assign(status=df["Churn_flag"].map({1: "Yes", 0: "No"}))
        ui.plot(donut(ch, "status", CHURN_COLORS, order=["No", "Yes"], title="Distribusi churn",
                      hole_text=f"{k['churn_rate']:.1f}%<br>churn"), key="ov_churn_donut")
    else:
        st.markdown("**Distribusi churn**")
        ui.note("Dataset unggahan tidak memiliki kolom Churn, sehingga distribusi churn aktual tidak tersedia. "
                "Lihat distribusi risiko prediksi di sebelah kanan.")
with c3:
    ui.plot(tier_bar(df, "risk_tier", RISK_ORDER, RISK_COLORS, title="Distribusi tingkat risiko"), key="ov_risk_bar")

with st.expander("Keterangan segmen"):
    for sid, name in df[["segment_id", "segment_name"]].drop_duplicates().sort_values("segment_id").itertuples(index=False):
        n = int((df["segment_id"] == sid).sum())
        st.markdown(f"- **{seg_short(name)}**: {name.split(':', 1)[1].strip()} ({n:,} pelanggan)")

# ------------------------------------------------------------------ Grafik baris 2
ui.section("Tagihan dan kontrak")
c4, c5 = st.columns([1.15, 1])
ch_label = df.assign(Churn=df["Churn_flag"].map({1: "Yes", 0: "No"})) if has_churn(df) else df.assign(Churn=None)
with c4:
    ui.plot(histogram(ch_label, "MonthlyCharges", "Churn", CHURN_COLORS, nbins=30, xtitle="Tagihan bulanan ($)",
                      title="Distribusi tagihan bulanan"), key="ov_hist")
with c5:
    ui.plot(stacked_count_bar(ch_label, "Contract", "Churn", CHURN_COLORS,
                              order=["Month-to-month", "One year", "Two year"], title="Distribusi jenis kontrak"),
            key="ov_contract")

# ------------------------------------------------------------------ Ringkasan analitik
ui.section("Ringkasan analitik", "Dihitung otomatis dari data yang sedang ditampilkan. Bersifat deskriptif (asosiasi), bukan sebab-akibat.")
with st.container(border=True):
    for p in executive_summary(df, table):
        st.markdown(f"- {p}")

with st.expander("Cara membaca tingkat risiko"):
    try:
        b = load_model_bundle()
        st.markdown(
            f"- **High**: probabilitas mentah model ≥ {b['risk_cut_high']:.3f} (kuantil ke-80 skor, yaitu 20% skor tertinggi).\n"
            f"- **Medium**: ≥ {b['risk_cut_med']:.3f} (kuantil ke-50).\n- **Low**: di bawahnya.")
    except ModelLoadError:
        st.markdown("- Cutoff risiko tidak dapat dimuat karena model tidak tersedia.")
    if info["mode"] == "default":
        ui.note("Untuk dataset bawaan, skor adalah prediksi <b>out-of-fold</b> dari notebook (setiap pelanggan diprediksi "
                "oleh model yang tidak pernah melihat dirinya), sehingga angka di sini sama dengan laporan analisis. "
                "Dengan definisi kuantil, jumlah risiko tinggi pada seluruh data selalu sekitar 20%; yang informatif adalah "
                "churn aktual per tingkat risiko pada grafik di atas.")
    else:
        ui.note("Untuk dataset unggahan, skor berasal dari model tersimpan dengan cutoff dari notebook. "
                "Proporsi tiap tingkat risiko bisa berbeda dari 20/30/50 jika populasi Anda berbeda.", warn=True)
