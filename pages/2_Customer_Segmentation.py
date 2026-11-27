"""Halaman 2: Customer Segmentation (hasil model segmentasi dari notebook)."""
import numpy as np
import pandas as pd
import streamlit as st

from utils import ui
from utils.analytics import (compute_kpis, has_churn, overall_profile, segment_narrative, segment_profile,
                             service_adoption)
from utils.config import RISK_ORDER
from utils.data_loader import load_report_csv
from utils.model_loader import ModelLoadError, load_model_bundle
from utils.visualizations import (churn_ci_bar, donut, heatmap, scatter_segments, seg_short, segment_bar,
                                  segment_color_map, stacked_pct_bar, CHURN_COLORS, TEAL, AMBER, VIOLET)

ui.bootstrap()
table, source_label, info = ui.get_dataset()
ui.page_header("Customer Segmentation",
               "Segmen pelanggan hasil model segmentasi yang dilatih di notebook: ukuran, profil, perilaku, dan churn.")
st.caption(f"Sumber data: {source_label}")

# ------------------------------------------------------------------ Filter
ui.section("Filter segmen", "Pilih segmen yang ingin dibandingkan. Proporsi tetap dihitung terhadap seluruh basis pelanggan.")
all_names = table.sort_values("segment_id")["segment_name"].unique().tolist()
sel = st.multiselect("Segmen", all_names, default=all_names, key="sg_sel")
if not sel:
    st.warning("Pilih minimal satu segmen.")
    st.stop()

prof_all = segment_profile(table)
ov = overall_profile(table)
prof = prof_all[prof_all["segment_name"].isin(sel)]
df = table[table["segment_name"].isin(sel)]
churn = has_churn(table)
cmap = segment_color_map(table)

try:
    b = load_model_bundle()
    method = str(b["final_method"]).upper().replace("KMEANS", "K-Means")
except ModelLoadError:
    method = "K-Means"

# ------------------------------------------------------------------ KPI
big = prof.sort_values("n", ascending=False).iloc[0]
items = [
    ("Jumlah cluster", f"{table['segment_id'].nunique()}", f"metode: {method}", "#0F6E8C"),
    ("Pelanggan dalam filter", f"{len(df):,}", f"{len(df) / len(table) * 100:.1f}% dari {len(table):,}", "#6C5B9E"),
    ("Segmen terbesar", seg_short(big["segment_name"]), f"{big['share_%']:.1f}% dari basis", "#F2A541"),
]
if churn:
    hi = prof.sort_values("churn_rate_%", ascending=False).iloc[0]
    items.append(("Churn tertinggi", f"{hi['churn_rate_%']:.1f}%", f"segmen {seg_short(hi['segment_name'])}", "#D1495B"))
else:
    items.append(("Churn tertinggi", "n/a", "data tanpa kolom Churn", "#D1495B"))
ui.kpi_row(items)

# ------------------------------------------------------------------ Ukuran
ui.section("Ukuran dan proporsi segmen")
c1, c2 = st.columns([1.2, 1])
with c1:
    ui.plot(segment_bar(prof, "n", "Jumlah pelanggan per segmen", fmt="{:,.0f}", ytitle="Pelanggan"), key="sg_size")
with c2:
    seg_df = df.assign(segmen=df["segment_name"].map(seg_short))
    ui.plot(donut(seg_df, "segmen", cmap, order=sorted(seg_df["segmen"].unique()), title="Proporsi dalam filter",
                  height=300), key="sg_donut")

# ------------------------------------------------------------------ Profil
ui.section("Profil segmen", "Setiap kolom adalah satu segmen. Nilai dihitung dari data yang sedang aktif.")
rows = [("Nama segmen", "segment_name", None), ("Jumlah pelanggan", "n", "{:,.0f}"), ("Proporsi basis (%)", "share_%", "{:.1f}"),
        ("Rata-rata tenure (bulan)", "tenure_mean", "{:.1f}"), ("Rata-rata tagihan bulanan ($)", "monthly_mean", "{:.2f}"),
        ("Rata-rata total tagihan ($)", "total_mean", "{:,.0f}"), ("Rata-rata layanan tambahan", "n_addon_mean", "{:.2f}"),
        ("Kontrak bulanan (%)", "pct_month_to_month", "{:.1f}"), ("Kontrak 1 tahun (%)", "pct_one_year", "{:.1f}"),
        ("Kontrak 2 tahun (%)", "pct_two_year", "{:.1f}"), ("Internet fiber optic (%)", "pct_fiber", "{:.1f}"),
        ("Internet DSL (%)", "pct_dsl", "{:.1f}"), ("Tanpa internet (%)", "pct_no_internet", "{:.1f}"),
        ("Electronic check (%)", "pct_echeck", "{:.1f}"), ("Paperless billing (%)", "pct_paperless", "{:.1f}"),
        ("Senior citizen (%)", "pct_senior", "{:.1f}"), ("Memiliki partner (%)", "pct_partner", "{:.1f}"),
        ("Memiliki tanggungan (%)", "pct_dependents", "{:.1f}"), ("Risiko tinggi (%)", "pct_high_risk", "{:.1f}")]
if churn:
    rows += [("Churn rate (%)", "churn_rate_%", "{:.1f}")]
tbl = {}
for sid, r in prof.iterrows():
    tbl[seg_short(r["segment_name"])] = [r[c] if f is None else f.format(r[c]) for _, c, f in rows]
prof_tbl = pd.DataFrame(tbl, index=[lab for lab, _, _ in rows])
if churn:
    prof_tbl.loc["CI 95% churn (%)"] = [f"{r['churn_ci_low_%']:.1f} - {r['churn_ci_high_%']:.1f}" for _, r in prof.iterrows()]
ui.dataframe(prof_tbl.astype(str))

# ------------------------------------------------------------------ Perbandingan
ui.section("Perbandingan metrik utama", "Garis putus-putus adalah rata-rata seluruh pelanggan.")
k1, k2, k3 = st.columns(3)
with k1:
    ui.plot(segment_bar(prof, "tenure_mean", "Rata-rata tenure (bulan)", fmt="{:.1f}", hline=ov["tenure_mean"]), key="sg_ten")
with k2:
    ui.plot(segment_bar(prof, "monthly_mean", "Rata-rata tagihan bulanan ($)", fmt="${:.2f}", hline=ov["monthly_mean"]), key="sg_mon")
with k3:
    ui.plot(segment_bar(prof, "total_mean", "Rata-rata total tagihan ($)", fmt="${:,.0f}", hline=ov["total_mean"]), key="sg_tot")

ui.section("Kontrak dan churn")
m1, m2 = st.columns(2)
with m1:
    cdf = df.assign(segmen=df["segment_name"].map(seg_short))
    ui.plot(stacked_pct_bar(cdf, "segmen", "Contract", {"Month-to-month": "#D1495B", "One year": AMBER, "Two year": TEAL},
                            x_order=sorted(cdf["segmen"].unique()), color_order=["Month-to-month", "One year", "Two year"],
                            title="Komposisi kontrak per segmen"), key="sg_contract")
with m2:
    if churn:
        ui.plot(churn_ci_bar(prof, ov["churn_rate_%"]), key="sg_churn")
    else:
        st.markdown("**Churn rate per segmen**")
        ui.note("Dataset tanpa kolom Churn: churn aktual per segmen tidak tersedia.")

# ------------------------------------------------------------------ Adopsi layanan
ui.section("Adopsi layanan per segmen",
           "Telepon dan internet: persentase dari semua pelanggan segmen. Layanan tambahan: persentase dari pelanggan internet "
           "di segmen tersebut (kosong jika kurang dari 30 pelanggan internet).")
ad = service_adoption(df)
ad.index = [seg_short(n) for n in prof_all.loc[ad.index, "segment_name"]]
ui.plot(heatmap(ad, text=ad, colorscale="Blues", fmt="{:.0f}%", height=60 + 70 * len(ad), colorbar_title="%"), key="sg_adopt")

# ------------------------------------------------------------------ Peta profil + scatter
ui.section("Peta profil segmen", "Warna menunjukkan posisi relatif antar segmen (z-score); angka adalah nilai sebenarnya.")
zcols = {"tenure_mean": "Tenure", "monthly_mean": "Tagihan", "n_addon_mean": "Add-on", "annual_value_mean": "Nilai tahunan",
         "pct_month_to_month": "% bulanan", "pct_fiber": "% fiber", "pct_echeck": "% e-check", "pct_senior": "% senior",
         "pct_partner": "% partner", "pct_dependents": "% tanggungan"}
if churn:
    zcols["churn_rate_%"] = "Churn %"
zsrc = prof_all[list(zcols)]
sd = zsrc.std(ddof=0).replace(0, np.nan)
z = ((zsrc - zsrc.mean()) / sd).fillna(0).loc[prof.index]
z.index = [seg_short(n) for n in prof["segment_name"]]
z.columns = list(zcols.values())
txt = zsrc.loc[prof.index].copy()
txt.index, txt.columns = z.index, z.columns
p1, p2 = st.columns([1.15, 1])
with p1:
    ui.plot(heatmap(z, text=txt, colorscale="RdBu_r", zmid=0, fmt="{:.1f}", height=300, title="Profil relatif antar segmen"), key="sg_z")
with p2:
    ui.plot(scatter_segments(df, "tenure", "MonthlyCharges", cmap, title="Tenure vs tagihan bulanan"), key="sg_scatter")

# ------------------------------------------------------------------ Interpretasi
ui.section("Interpretasi tiap segmen",
           "Ringkasan dibuat otomatis dari angka segmen dibandingkan rata-rata keseluruhan. Ini asosiasi, bukan sebab-akibat.")
recs = load_report_csv("business/business_recommendations.csv") if info["mode"] == "default" else None
tabs = st.tabs([seg_short(n) for n in prof["segment_name"]])
for tab, (sid, r) in zip(tabs, prof.iterrows()):
    with tab:
        st.markdown(f"#### {r['segment_name']}")
        a, b_, c_ = st.columns(3)
        a.metric(f"Pelanggan ({r['share_%']:.1f}% basis)", f"{int(r['n']):,}")
        b_.metric("Churn rate", f"{r['churn_rate_%']:.1f}%" if churn else "n/a",
                  f"{r['churn_rate_%'] - ov['churn_rate_%']:+.1f} poin vs rata-rata" if churn else None, delta_color="inverse")
        c_.metric("Tenure rata-rata", f"{r['tenure_mean']:.1f} bln", f"{r['tenure_mean'] - ov['tenure_mean']:+.1f} vs rata-rata",
                  delta_color="off")
        for line in segment_narrative(r, ov):
            st.markdown(f"- {line}")
        if recs is not None:
            hit = recs[recs["segment_id"] == sid]
            if len(hit):
                h = hit.iloc[0]
                st.markdown("**Rekomendasi dari analisis notebook** *(hipotesis berbasis asosiasi, perlu diuji A/B)*")
                for act in str(h["rekomendasi"]).split(" | "):
                    st.markdown(f"- {act}")
                st.caption(f"Bukti data: {h['bukti_data']}")
        else:
            st.caption("Rekomendasi bisnis dari notebook hanya tersedia untuk dataset bawaan. "
                       "Lihat halaman Retention Strategy untuk detailnya.")

with st.expander("Bagaimana segmen dibentuk dan diberi nama"):
    try:
        feats = ", ".join(load_model_bundle()["_cf"].CLUSTER_FEATS)
    except ModelLoadError:
        feats = "tenure, MonthlyCharges, n_addon_services, contract_ord"
    t33, t67 = table["tenure"].quantile([1 / 3, 2 / 3])
    m33, m67 = table["MonthlyCharges"].quantile([1 / 3, 2 / 3])
    st.markdown(
        f"- **Metode:** {method} dengan {table['segment_id'].nunique()} cluster pada fitur terstandardisasi: `{feats}`. "
        "Churn **tidak** dipakai untuk membentuk cluster; churn hanya dipakai setelahnya untuk evaluasi.\n"
        "- **ID segmen** diurutkan dari tenure rata-rata terendah ke tertinggi.\n"
        f"- **Nama segmen** dibuat otomatis oleh aturan di notebook, bukan oleh analis. Label tenure: *Baru* jika rata-rata tenure "
        f"segmen < {t33:.0f} bulan, *Loyal* jika > {t67:.0f} bulan, selain itu *Menengah*. Label tagihan: *rendah* jika < ${m33:.0f}, "
        f"*tinggi* jika > ${m67:.0f}, selain itu *sedang*. Batas ini adalah tercile seluruh pelanggan.")
    mid = prof_all[prof_all["segment_name"].str.contains("Menengah")]
    if len(mid) > 1:
        lst = ", ".join(f"{seg_short(r['segment_name'])} ({r['tenure_mean']:.0f} bln)" for _, r in mid.iterrows())
        ui.note(f"Perhatian: label <b>Menengah</b> mencakup rentang tenure yang luas ({lst}). Gunakan angka profil, bukan hanya nama, "
                "saat menafsirkan segmen.", warn=True)
    st.caption("Nama segmen tidak diubah di aplikasi agar konsisten dengan hasil notebook.")
