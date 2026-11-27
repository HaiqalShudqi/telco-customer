"""Halaman 6: Retention Strategy (prioritas retensi dan rekomendasi berbasis bukti)."""
import numpy as np
import pandas as pd
import streamlit as st

from utils import ui
from utils.analytics import has_churn, overall_profile, segment_profile
from utils.business import PRIORITY_ACTIONS, PRIORITY_DEFINITION, REVENUE_AT_RISK_NOTE
from utils.config import PRIORITY_ORDER, RISK_ORDER
from utils.data_loader import load_report_csv
from utils.visualizations import (PRIORITY_COLORS, RISK_COLORS, SLATE, heatmap, hbar, seg_short, segment_bar, style_fig)

import plotly.graph_objects as go

ui.bootstrap()
table, source_label, info = ui.get_dataset()
churn = has_churn(table)
ui.page_header("Retention Strategy",
               "Mengubah hasil analisis menjadi tindakan: siapa yang diprioritaskan, seberapa besar eksposurnya, dan apa langkahnya.")
st.caption(f"Sumber data: {source_label}")

# ------------------------------------------------------------------ Filter
ui.section("Filter")
seg_names = table.sort_values("segment_id")["segment_name"].unique().tolist()
risks = [r for r in RISK_ORDER if r in table["risk_tier"].unique()]
prios = [p for p in PRIORITY_ORDER if p in table["priority"].unique()]
values = [v for v in ["High", "Standard"] if v in table["value_tier"].unique()]
f1, f2 = st.columns([2.2, 1.4])
sel_seg = f1.multiselect("Segmen", seg_names, default=seg_names, key="rs_seg")
sel_prio = f2.multiselect("Prioritas retensi", prios, default=prios, key="rs_prio")
f3, f4 = st.columns(2)
sel_risk = f3.multiselect("Tingkat risiko", risks, default=risks, key="rs_risk")
sel_val = f4.multiselect("Tingkat nilai pelanggan", values, default=values, key="rs_val")

df = table[table["segment_name"].isin(sel_seg) & table["priority"].isin(sel_prio) & table["risk_tier"].isin(sel_risk)
           & table["value_tier"].isin(sel_val)]
ui.stop_if_empty(df)

# ------------------------------------------------------------------ KPI
total_rar = table["revenue_at_risk"].sum()
rar = df["revenue_at_risk"].sum()
n_p1 = int(df["priority"].str.startswith("P1").sum())
ui.kpi_row([
    ("Pelanggan dalam filter", f"{len(df):,}", f"dari {len(table):,}" if len(df) != len(table) else "seluruh basis", "#0F6E8C"),
    ("Risiko tinggi", f"{int((df['risk_tier'] == 'High').sum()):,}", f"{(df['risk_tier'] == 'High').mean() * 100:.1f}% dari filter", "#D1495B"),
    ("Nilai tinggi", f"{int((df['value_tier'] == 'High').sum()):,}", f"{(df['value_tier'] == 'High').mean() * 100:.1f}% dari filter", "#14213D"),
    ("Prioritas P1", f"{n_p1:,}", "risiko tinggi dan nilai tinggi", "#F2A541"),
    ("Revenue at risk (proksi)", f"{rar:,.0f}", f"{rar / total_rar * 100:.1f}% dari total basis" if total_rar else "", "#6C5B9E"),
])
st.write("")
ui.note(REVENUE_AT_RISK_NOTE, warn=True)

# ------------------------------------------------------------------ Distribusi prioritas
ui.section("Distribusi prioritas retensi",
           "Jumlah pelanggan dan porsi revenue at risk (proksi) pada tiap prioritas, dalam filter yang aktif.")
pt = df.groupby("priority").agg(n=("customerID", "size"), rar=("revenue_at_risk", "sum")).reindex(PRIORITY_ORDER).fillna(0)
pt["share_n"] = pt["n"] / pt["n"].sum() * 100
pt["share_rar"] = pt["rar"] / pt["rar"].sum() * 100 if pt["rar"].sum() else 0
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure(go.Bar(x=[p.split(" - ")[0] for p in pt.index], y=pt["n"], marker_color=[PRIORITY_COLORS[p] for p in pt.index],
                           text=[f"{int(n):,}<br>{s:.1f}%" for n, s in zip(pt["n"], pt["share_n"])], textposition="outside",
                           cliponaxis=False, hovertext=pt.index, hovertemplate="%{hovertext}<br>%{y:,} pelanggan<extra></extra>"))
    fig.update_yaxes(title="Pelanggan", range=[0, max(pt["n"].max(), 1) * 1.22])
    ui.plot(style_fig(fig, 320, False, "Jumlah pelanggan per prioritas"), key="rs_prio_n")
with c2:
    fig = go.Figure(go.Bar(x=[p.split(" - ")[0] for p in pt.index], y=pt["share_rar"], marker_color=[PRIORITY_COLORS[p] for p in pt.index],
                           text=[f"{v:.1f}%" for v in pt["share_rar"]], textposition="outside", cliponaxis=False,
                           hovertext=pt.index, hovertemplate="%{hovertext}<br>%{y:.1f}% dari revenue at risk<extra></extra>"))
    fig.update_yaxes(title="% revenue at risk (proksi)", range=[0, max(float(pt["share_rar"].max()), 1) * 1.22])
    ui.plot(style_fig(fig, 320, False, "Porsi revenue at risk (proksi) per prioritas"), key="rs_prio_rar")

# Matriks risiko x nilai (label prioritas diambil dari data, bukan dihitung ulang)
cnt = pd.crosstab(df["risk_tier"], df["value_tier"]).reindex(index=[r for r in ["High", "Medium", "Low"] if r in df["risk_tier"].unique()],
                                                              columns=[v for v in ["High", "Standard"] if v in df["value_tier"].unique()]).fillna(0)
lab = cnt.copy().astype(object)
for r_ in cnt.index:
    for v_ in cnt.columns:
        sub = df[(df["risk_tier"] == r_) & (df["value_tier"] == v_)]["priority"]
        lab.loc[r_, v_] = f"{int(cnt.loc[r_, v_]):,}<br>{sub.mode().iloc[0].split(' - ')[0]}" if len(sub) else "0"
fig = go.Figure(go.Heatmap(z=cnt.values, x=[f"Nilai {c}" for c in cnt.columns], y=[f"Risiko {i}" for i in cnt.index], text=lab.values,
                           texttemplate="%{text}", colorscale="Blues", xgap=3, ygap=3, showscale=False,
                           hovertemplate="%{y} | %{x}<br>%{z:,.0f} pelanggan<extra></extra>"))
fig.update_yaxes(autorange="reversed")
ui.plot(style_fig(fig, 260, False, "Matriks risiko × nilai (angka = pelanggan; kode = prioritas)"), key="rs_matrix")

# Ringkasan per prioritas
g = df.groupby("priority")
summary = pd.DataFrame({
    "Pelanggan": g.size(), "Porsi (%)": g.size() / len(df) * 100,
    "Churn aktual (%)": g["Churn_flag"].mean() * 100 if churn else np.nan,
    "Prob. churn rata-rata (%)": g["churn_prob_cal"].mean() * 100,
    "Tenure median (bln)": g["tenure"].median(), "Tagihan rata-rata ($)": g["MonthlyCharges"].mean(),
    "Nilai tahunan total (proksi)": g["annual_value_proxy"].sum(), "Revenue at risk (proksi)": g["revenue_at_risk"].sum(),
}).reindex([p for p in PRIORITY_ORDER if p in g.groups])
summary["Porsi revenue at risk (%)"] = summary["Revenue at risk (proksi)"] / summary["Revenue at risk (proksi)"].sum() * 100
summary.insert(0, "Definisi", [PRIORITY_DEFINITION[p] for p in summary.index])
ui.dataframe(summary.round(1))
if not churn:
    st.caption("Churn aktual tidak tersedia karena dataset tidak memiliki kolom Churn.")

# ------------------------------------------------------------------ Panduan tindakan
ui.section("Panduan tindakan per prioritas", "Berbasis aturan prioritas, bukan keluaran model. Uji efeknya dengan A/B test.")
for p in PRIORITY_ORDER:
    if p in prios:
        with st.container(border=True):
            st.markdown(f"**{p}**  \n*{PRIORITY_DEFINITION[p]}*")
            st.markdown(PRIORITY_ACTIONS[p])

# ------------------------------------------------------------------ Segmen
ui.section("Rekomendasi retensi per segmen", "Eksposur tiap segmen dihitung dari data yang sedang ditampilkan.")
prof = segment_profile(df)
ov = overall_profile(table)
sg = df.groupby("segment_id").agg(n=("customerID", "size"), prob=("churn_prob_cal", "mean"), rar=("revenue_at_risk", "sum"),
                                  p1=("priority", lambda s: s.str.startswith("P1").mean() * 100))
sg["share_rar"] = sg["rar"] / sg["rar"].sum() * 100
seg_tbl = pd.DataFrame({
    "Segmen": [prof.loc[i, "segment_name"] for i in sg.index], "Pelanggan": sg["n"],
    "Churn aktual (%)": [prof.loc[i, "churn_rate_%"] for i in sg.index], "Prob. churn rata-rata (%)": sg["prob"] * 100,
    "P1 (% segmen)": sg["p1"], "Porsi revenue at risk (%)": sg["share_rar"], "Tenure rata-rata (bln)": [prof.loc[i, "tenure_mean"] for i in sg.index],
    "Tagihan rata-rata ($)": [prof.loc[i, "monthly_mean"] for i in sg.index]})
ui.dataframe(seg_tbl.round(1), hide_index=True)

recs = load_report_csv("business/business_recommendations.csv") if info["mode"] == "default" else None
if recs is None:
    ui.note("Rekomendasi per segmen dari notebook hanya tersedia untuk dataset bawaan. Untuk dataset unggahan, gunakan tabel eksposur di atas "
            "dan panduan tindakan per prioritas.")
else:
    tabs = st.tabs([seg_short(n) for n in prof["segment_name"]])
    for tab, (sid, row) in zip(tabs, prof.iterrows()):
        with tab:
            h = recs[recs["segment_id"] == sid]
            st.markdown(f"#### {row['segment_name']}")
            if len(h) == 0:
                st.info("Tidak ada rekomendasi tersimpan untuk segmen ini.")
                continue
            h = h.iloc[0]
            st.markdown(f"Tingkat risiko segmen menurut notebook: **{h['tingkat_risiko_segmen']}**. "
                        f"Dalam filter saat ini, segmen ini memuat **{sg.loc[sid, 'share_rar']:.1f}%** revenue at risk (proksi) dan "
                        f"**{sg.loc[sid, 'p1']:.1f}%** pelanggannya berprioritas P1.")
            st.markdown("**Rekomendasi**")
            for act in str(h["rekomendasi"]).split(" | "):
                st.markdown(f"- {act}")
            st.caption(f"Bukti data: {h['bukti_data']}")
            st.caption(str(h["catatan"]))

# ------------------------------------------------------------------ Tabel prioritisasi
ui.section("Daftar prioritisasi pelanggan", "Diurutkan menurut revenue at risk (proksi), dari yang terbesar.")
lst = df.sort_values(["revenue_at_risk", "customerID"], ascending=[False, True]).reset_index(drop=True)
view = pd.DataFrame({
    "customerID": lst["customerID"], "Segmen": lst["segment_name"].map(seg_short), "Prioritas": lst["priority"],
    "Risiko": lst["risk_tier"], "Nilai": lst["value_tier"], "Prob. churn (%)": lst["churn_prob_cal"] * 100,
    "Revenue at risk (proksi)": lst["revenue_at_risk"], "Tenure": lst["tenure"], "Kontrak": lst["Contract"],
    "Internet": lst["InternetService"], "Pembayaran": lst["PaymentMethod"], "Tagihan/bln ($)": lst["MonthlyCharges"],
    "Layanan tambahan": lst["n_addon_services"],
})
ui.dataframe(view, hide_index=True, height=420, column_config={
    "Prob. churn (%)": st.column_config.ProgressColumn("Prob. churn (terkalibrasi)", format="%.1f%%", min_value=0, max_value=100),
    "Revenue at risk (proksi)": st.column_config.NumberColumn(format="%.1f"),
    "Tagihan/bln ($)": st.column_config.NumberColumn(format="$%.2f")})

EXPORT = ["customerID", "segment_id", "segment_name", "priority", "risk_tier", "value_tier", "churn_prob_raw", "churn_prob_cal",
          "revenue_at_risk", "annual_value_proxy", "tenure", "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
          "MonthlyCharges", "TotalCharges", "n_addon_services"] + (["Churn"] if churn else [])
export = lst[[c for c in EXPORT if c in lst.columns]]
st.download_button("Unduh daftar prioritisasi (CSV)", export.to_csv(index=False).encode("utf-8"),
                   file_name="telco_retention_priority_list.csv", mime="text/csv", key="rs_dl")
st.caption(f"{len(export):,} pelanggan akan diunduh sesuai filter aktif. Daftar ini alat bantu prioritisasi, bukan keputusan otomatis.")
