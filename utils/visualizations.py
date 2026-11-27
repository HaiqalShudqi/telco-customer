"""Grafik Plotly dengan identitas visual yang konsisten di seluruh halaman."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NAVY = "#14213D"
TEAL = "#0F6E8C"
CORAL = "#D1495B"
AMBER = "#F2A541"
GREEN = "#2E933C"
VIOLET = "#6C5B9E"
SLATE = "#64748B"
GRID = "#E5E9F0"

FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

SEGMENT_COLORS = {0: TEAL, 1: AMBER, 2: VIOLET, 3: GREEN, 4: SLATE, 5: CORAL}
RISK_COLORS = {"Low": GREEN, "Medium": AMBER, "High": CORAL}
CHURN_COLORS = {"No": TEAL, "Yes": CORAL}
VALUE_COLORS = {"Standard": SLATE, "High": NAVY}
PRIORITY_COLORS = {
    "P1 - Selamatkan segera": CORAL,
    "P2 - Retensi terstruktur": AMBER,
    "P3 - Pantau": TEAL,
    "P4 - Pertahankan": GREEN,
}


def seg_short(name: str) -> str:
    """'S0: Menengah | ...' -> 'S0' (label ringkas untuk sumbu dan legenda grafik)."""
    return str(name).split(":")[0].strip()


def segment_color_map(table: pd.DataFrame) -> dict[str, str]:
    """Peta warna berdasarkan label pendek segmen, stabil antar halaman."""
    out = {}
    for sid, name in table[["segment_id", "segment_name"]].drop_duplicates().itertuples(index=False):
        out[seg_short(name)] = SEGMENT_COLORS.get(int(sid), SLATE)
    return out


def style_fig(fig: go.Figure, height: int = 340, legend: bool = True, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=44 if title else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=NAVY),
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=15, color=NAVY)) if title else None,
        showlegend=bool(legend),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0, font=dict(size=11), title=None),
        hoverlabel=dict(font_family=FONT, bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def donut(df: pd.DataFrame, col: str, color_map: dict, order: list | None = None, title: str | None = None,
          height: int = 320, hole_text: str | None = None) -> go.Figure:
    counts = df[col].value_counts()
    labels = [x for x in (order or counts.index.tolist()) if x in counts.index]
    values = [int(counts[x]) for x in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62, sort=False,
        marker=dict(colors=[color_map.get(x, SLATE) for x in labels], line=dict(color="white", width=2)),
        textinfo="percent", textfont=dict(size=12, color="white"),
        hovertemplate="%{label}<br>%{value:,} pelanggan (%{percent})<extra></extra>"))
    if hole_text:
        fig.add_annotation(text=hole_text, x=0.5, y=0.5, showarrow=False, font=dict(size=15, color=NAVY))
    return style_fig(fig, height, True, title)


def stacked_count_bar(df: pd.DataFrame, x: str, color: str | None, color_map: dict, order: list | None = None,
                      title: str | None = None, height: int = 320) -> go.Figure:
    """Jumlah pelanggan per kategori x; ditumpuk menurut `color` jika diberikan."""
    if color and df[color].notna().any():
        g = df.groupby([x, color], observed=True).size().reset_index(name="n")
        fig = px.bar(g, x=x, y="n", color=color, color_discrete_map=color_map,
                     category_orders={x: order} if order else None, barmode="stack")
    else:
        g = df.groupby(x, observed=True).size().reset_index(name="n")
        fig = px.bar(g, x=x, y="n", category_orders={x: order} if order else None)
        fig.update_traces(marker_color=TEAL)
    fig.update_traces(hovertemplate="%{x}<br>%{y:,} pelanggan<extra></extra>")
    fig.update_yaxes(title="Pelanggan")
    fig.update_xaxes(title=None)
    return style_fig(fig, height, bool(color and df[color].notna().any()), title)


def histogram(df: pd.DataFrame, col: str, color: str | None, color_map: dict, nbins: int = 30,
              title: str | None = None, height: int = 320, xtitle: str | None = None) -> go.Figure:
    use_color = bool(color and df[color].notna().any())
    fig = px.histogram(df, x=col, color=color if use_color else None, nbins=nbins, color_discrete_map=color_map,
                       barmode="overlay", opacity=0.72)
    if not use_color:
        fig.update_traces(marker_color=TEAL)
    fig.update_traces(hovertemplate=f"{xtitle or col}: %{{x}}<br>%{{y:,}} pelanggan<extra></extra>")
    fig.update_xaxes(title=xtitle or col)
    fig.update_yaxes(title="Pelanggan")
    return style_fig(fig, height, use_color, title)


def tier_bar(table: pd.DataFrame, tier_col: str, order: list, color_map: dict, title: str | None = None,
             height: int = 320) -> go.Figure:
    """Jumlah pelanggan per tier; label menampilkan churn aktual jika tersedia (validasi tier)."""
    has_churn = table["Churn_flag"].notna().any()
    rows = []
    for t in order:
        s = table[table[tier_col] == t]
        label = f"{len(s):,}"
        if has_churn and len(s):
            label += f"<br>churn aktual {s['Churn_flag'].mean() * 100:.1f}%"
        rows.append((t, len(s), label))
    fig = go.Figure(go.Bar(
        x=[r[0] for r in rows], y=[r[1] for r in rows], text=[r[2] for r in rows], textposition="outside",
        marker_color=[color_map.get(r[0], SLATE) for r in rows], cliponaxis=False,
        hovertemplate="%{x}<br>%{y:,} pelanggan<extra></extra>"))
    fig.update_yaxes(title="Pelanggan", range=[0, max(r[1] for r in rows) * 1.22 if rows else 1])
    return style_fig(fig, height, False, title)


# ==============================================================================================
# Stage 4
# ==============================================================================================
def segment_bar(prof: pd.DataFrame, col: str, title: str, fmt: str = "{:.1f}", ytitle: str = "",
                height: int = 300, hline: float | None = None, hline_label: str = "rata-rata keseluruhan") -> go.Figure:
    """Satu batang per segmen (warna segmen, label ringkas S0/S1/...)."""
    labels = [seg_short(n) for n in prof["segment_name"]]
    colors = [SEGMENT_COLORS.get(int(i), SLATE) for i in prof.index]
    fig = go.Figure(go.Bar(x=labels, y=prof[col], marker_color=colors, text=[fmt.format(v) for v in prof[col]],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>" + title + ": %{y:,.2f}<extra></extra>"))
    if hline is not None and not pd.isna(hline):
        fig.add_hline(y=hline, line_dash="dash", line_color=NAVY, line_width=1,
                      annotation_text=f"{hline_label}: {fmt.format(hline)}", annotation_position="top left",
                      annotation_font_size=10)
    top = float(prof[col].max()) if len(prof) else 1
    fig.update_yaxes(title=ytitle, range=[0, max(top, hline or 0) * 1.25 or 1])
    return style_fig(fig, height, False, title)


def churn_ci_bar(prof: pd.DataFrame, overall: float, title: str = "Churn rate per segmen (CI 95%)",
                 height: int = 320) -> go.Figure:
    labels = [seg_short(n) for n in prof["segment_name"]]
    colors = [SEGMENT_COLORS.get(int(i), SLATE) for i in prof.index]
    fig = go.Figure(go.Bar(
        x=labels, y=prof["churn_rate_%"], marker_color=colors, cliponaxis=False,
        error_y=dict(type="data", symmetric=False, array=prof["churn_ci_high_%"] - prof["churn_rate_%"],
                     arrayminus=prof["churn_rate_%"] - prof["churn_ci_low_%"], color=NAVY, thickness=1.2),
        text=[f"{v:.1f}%" for v in prof["churn_rate_%"]], textposition="outside",
        hovertemplate="%{x}<br>churn %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dash", line_color=NAVY, line_width=1,
                  annotation_text=f"rata-rata {overall:.1f}%", annotation_position="top left", annotation_font_size=10)
    fig.update_yaxes(title="Churn rate (%)", range=[0, max(float(prof["churn_ci_high_%"].max()), overall) * 1.3])
    return style_fig(fig, height, False, title)


def stacked_pct_bar(df: pd.DataFrame, x: str, color: str, color_map: dict, x_order: list | None = None,
                    color_order: list | None = None, title: str | None = None, height: int = 320,
                    x_labels: dict | None = None) -> go.Figure:
    """Komposisi persen (100%) per kategori x."""
    ct = pd.crosstab(df[x], df[color], normalize="index") * 100
    if x_order:
        ct = ct.reindex([o for o in x_order if o in ct.index])
    cols = [c for c in (color_order or ct.columns.tolist()) if c in ct.columns]
    fig = go.Figure()
    xs = [(x_labels or {}).get(i, i) for i in ct.index]
    for c in cols:
        fig.add_trace(go.Bar(name=str(c), x=xs, y=ct[c], marker_color=color_map.get(c, SLATE),
                             text=[f"{v:.0f}%" if v >= 6 else "" for v in ct[c]], textposition="inside",
                             textfont=dict(color="white", size=11),
                             hovertemplate="%{x}<br>" + str(c) + ": %{y:.1f}%<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="% pelanggan", range=[0, 100])
    return style_fig(fig, height, True, title)


def heatmap(z: pd.DataFrame, title: str | None = None, text: pd.DataFrame | None = None, colorscale: str = "Blues",
            zmid: float | None = None, height: int = 340, fmt: str = "{:.1f}", colorbar_title: str | None = None) -> go.Figure:
    txt = text if text is not None else z
    labels = [[("" if pd.isna(v) else fmt.format(v)) for v in row] for row in txt.values]
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[str(c) for c in z.columns], y=[str(i) for i in z.index], text=labels, texttemplate="%{text}",
        colorscale=colorscale, zmid=zmid, xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(title=colorbar_title, thickness=12, len=0.8),
        hovertemplate="%{y} | %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(side="bottom", tickangle=0)
    return style_fig(fig, height, False, title)


def rate_bar(rt: pd.DataFrame, col: str, overall: float, title: str | None = None, height: int = 300,
             labels: dict | None = None) -> go.Figure:
    """Churn rate per kategori dengan label n dan garis rata-rata. rt berasal dari analytics.rate_table."""
    xs = [(labels or {}).get(str(v), str(v)) for v in rt[col]]
    fig = go.Figure(go.Bar(
        x=xs, y=rt["churn_rate"], marker_color=[CORAL if r > overall else TEAL for r in rt["churn_rate"]],
        text=[f"{r:.1f}%<br>n={n:,}" for r, n in zip(rt["churn_rate"], rt["n"])], textposition="outside", cliponaxis=False,
        error_y=dict(type="data", symmetric=False, array=rt["ci_high"] - rt["churn_rate"],
                     arrayminus=rt["churn_rate"] - rt["ci_low"], color=SLATE, thickness=1),
        hovertemplate="%{x}<br>churn %{y:.1f}% (CI 95% sesuai garis)<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dash", line_color=NAVY, line_width=1,
                  annotation_text=f"rata-rata {overall:.1f}%", annotation_position="top right", annotation_font_size=10)
    fig.update_yaxes(title="Churn rate (%)", range=[0, max(float(rt["ci_high"].max()), overall) * 1.35])
    return style_fig(fig, height, False, title)


def grouped_bar(df: pd.DataFrame, x: str, y: str, color: str, color_map: dict, title: str | None = None,
                height: int = 330, ytitle: str = "Churn rate (%)", fmt: str = "{:.1f}%") -> go.Figure:
    fig = go.Figure()
    for c in df[color].unique():
        s = df[df[color] == c]
        fig.add_trace(go.Bar(name=str(c), x=s[x], y=s[y], marker_color=color_map.get(c, SLATE),
                             text=[fmt.format(v) for v in s[y]], textposition="outside", cliponaxis=False))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=ytitle, range=[0, float(df[y].max()) * 1.25])
    return style_fig(fig, height, True, title)


def line_chart(series: pd.Series, overall: float, title: str, xtitle: str, ytitle: str, height: int = 320) -> go.Figure:
    fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines", line=dict(color=CORAL, width=2.5),
                               hovertemplate=xtitle + " %{x}<br>%{y:.1f}%<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dash", line_color=NAVY, line_width=1,
                  annotation_text=f"rata-rata {overall:.1f}%", annotation_position="top right", annotation_font_size=10)
    fig.update_xaxes(title=xtitle); fig.update_yaxes(title=ytitle, rangemode="tozero")
    return style_fig(fig, height, False, title)


def hbar(labels: list, values: list, colors: list | str, title: str | None = None, height: int = 380,
         xtitle: str = "", fmt: str = "{:.3f}") -> go.Figure:
    fig = go.Figure(go.Bar(y=labels, x=values, orientation="h", marker_color=colors,
                           text=[fmt.format(v) for v in values], textposition="outside", cliponaxis=False))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title=xtitle, range=[0, max(values) * 1.18 if len(values) else 1])
    return style_fig(fig, height, False, title)


def forest(or_df: pd.DataFrame, title: str | None = None, height: int = 520) -> go.Figure:
    """Odds ratio dengan CI 95% (sumbu log). or_df berkolom: label, odds_ratio, ci_low, ci_high, sig."""
    d = or_df.reset_index(drop=True)
    fig = go.Figure(go.Scatter(
        x=d["odds_ratio"], y=d["label"], mode="markers",
        marker=dict(size=9, color=[CORAL if s else SLATE for s in d["sig"]]),
        error_x=dict(type="data", symmetric=False, array=d["ci_high"] - d["odds_ratio"],
                     arrayminus=d["odds_ratio"] - d["ci_low"], color=SLATE, thickness=1.2),
        hovertemplate="%{y}<br>OR %{x:.2f}<extra></extra>"))
    fig.add_vline(x=1, line_dash="dash", line_color=NAVY, line_width=1)
    fig.update_xaxes(type="log", title="Odds ratio (skala log; 1 = tanpa asosiasi)")
    fig.update_yaxes(autorange="reversed", showgrid=True, gridcolor=GRID)
    return style_fig(fig, height, False, title)


def scatter_segments(df: pd.DataFrame, x: str, y: str, color_map: dict, title: str | None = None, height: int = 360,
                     sample: int = 3000, seed: int = 42) -> go.Figure:
    d = df.sample(min(sample, len(df)), random_state=seed).assign(segmen=lambda t: t["segment_name"].map(seg_short))
    fig = px.scatter(d, x=x, y=y, color="segmen", color_discrete_map=color_map, opacity=0.55,
                     category_orders={"segmen": sorted(d["segmen"].unique())})
    fig.update_traces(marker=dict(size=6))
    return style_fig(fig, height, True, title)


def prob_gauge(prob_pct: float, color: str, title: str = "Probabilitas churn (terkalibrasi)", height: int = 260) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob_pct, number=dict(suffix="%", font=dict(size=40, color=NAVY)),
        title=dict(text=title, font=dict(size=14, color=NAVY)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticktext=["0", "25", "50", "75", "100"]), bar=dict(color=color, thickness=0.35),
                   bgcolor="#EEF1F6", borderwidth=0)))
    fig.update_layout(height=height, margin=dict(l=45, r=45, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT))
    return fig


def risk_zone_bar(score: float, cut_med: float, cut_high: float, height: int = 150) -> go.Figure:
    """Posisi skor mentah model terhadap zona Low / Medium / High."""
    zones = [("Low", 0, cut_med, GREEN), ("Medium", cut_med, cut_high, AMBER), ("High", cut_high, 1, CORAL)]
    fig = go.Figure()
    for name, lo, hi, col in zones:
        fig.add_trace(go.Bar(y=["Skor"], x=[hi - lo], base=lo, orientation="h", name=name, marker_color=col, opacity=0.85,
                             text=name, textposition="inside", insidetextanchor="start", textfont=dict(color="white"),
                             hovertemplate=f"{name}: {lo:.3f} - {hi:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[score], y=["Skor"], mode="markers", name="Pelanggan ini",
                             marker=dict(symbol="diamond", size=18, color=NAVY, line=dict(color="white", width=2)),
                             hovertemplate=f"Skor mentah: {score:.3f}<extra></extra>"))
    fig.update_xaxes(range=[0, 1], title="Skor mentah model (batas zona dari notebook)")
    fig.update_yaxes(visible=False)
    fig.update_layout(barmode="overlay")
    return style_fig(fig, height, False, None)


# ==============================================================================================
# Stage 6: evaluasi model dan metodologi
# ==============================================================================================
MODEL_COLORS = [TEAL, CORAL, AMBER, VIOLET, GREEN, SLATE, "#0B3C5D"]


def roc_figure(curves: dict, highlight: str | None = None, height: int = 380) -> go.Figure:
    """curves: nama -> (fpr, tpr, auc)."""
    fig = go.Figure()
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        hl = name == highlight
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})",
                                 line=dict(width=3.5 if hl else 1.6, color=MODEL_COLORS[i % len(MODEL_COLORS)]),
                                 opacity=1 if (hl or highlight is None) else 0.55))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=SLATE), name="Acak", showlegend=False))
    fig.update_xaxes(title="False positive rate", range=[0, 1]); fig.update_yaxes(title="True positive rate", range=[0, 1.01])
    style_fig(fig, height, True, "Kurva ROC (test set)")
    fig.update_layout(legend=dict(orientation="v", y=0.04, x=0.98, xanchor="right", yanchor="bottom", font=dict(size=10),
                                  bgcolor="rgba(255,255,255,0.85)"))
    return fig


def pr_figure(curves: dict, baseline: float, highlight: str | None = None, height: int = 380) -> go.Figure:
    """curves: nama -> (recall, precision, ap)."""
    fig = go.Figure()
    for i, (name, (rec, prec, ap)) in enumerate(curves.items()):
        hl = name == highlight
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{name} (AP {ap:.3f})",
                                 line=dict(width=3.5 if hl else 1.6, color=MODEL_COLORS[i % len(MODEL_COLORS)]),
                                 opacity=1 if (hl or highlight is None) else 0.55))
    fig.add_hline(y=baseline, line_dash="dot", line_color=SLATE, annotation_text=f"baseline {baseline:.3f}",
                  annotation_position="bottom left", annotation_font_size=10)
    fig.update_xaxes(title="Recall", range=[0, 1]); fig.update_yaxes(title="Precision", range=[0, 1.01])
    style_fig(fig, height, True, "Kurva Precision-Recall (test set)")
    fig.update_layout(legend=dict(orientation="v", y=0.12, x=0.02, xanchor="left", yanchor="bottom", font=dict(size=10),
                                  bgcolor="rgba(255,255,255,0.85)"))
    return fig


def confusion_figure(tn: int, fp: int, fn: int, tp: int, title: str | None = None, height: int = 330) -> go.Figure:
    z = [[tn, fp], [fn, tp]]
    total = tn + fp + fn + tp
    text = [[f"{v:,}<br>{v / total * 100:.1f}%" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=["Prediksi: Tidak churn", "Prediksi: Churn"], y=["Aktual: Tidak churn", "Aktual: Churn"],
                               text=text, texttemplate="%{text}", colorscale="Blues", showscale=False, xgap=3, ygap=3,
                               textfont=dict(size=15)))
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height, False, title)


def threshold_figure(tbl: pd.DataFrame, thr: float, height: int = 330) -> go.Figure:
    fig = go.Figure()
    for col, name, color in (("precision", "Precision", TEAL), ("recall", "Recall", CORAL), ("f1", "F1", AMBER), ("f_beta", "F2", VIOLET)):
        fig.add_trace(go.Scatter(x=tbl["threshold"], y=tbl[col], mode="lines", name=name, line=dict(color=color, width=2.2)))
    fig.add_vline(x=thr, line_dash="dash", line_color=NAVY, annotation_text=f"threshold {thr:.2f}", annotation_font_size=10)
    fig.update_xaxes(title="Threshold"); fig.update_yaxes(title="Nilai", range=[0, 1.02])
    return style_fig(fig, height, True, "Trade-off precision / recall menurut threshold")


def multi_line(df: pd.DataFrame, x: str, ys: list[tuple[str, str]], mark_x=None, title: str | None = None,
               height: int = 280, ytitle: str = "") -> go.Figure:
    fig = go.Figure()
    for i, (col, name) in enumerate(ys):
        fig.add_trace(go.Scatter(x=df[x], y=df[col], mode="lines+markers", name=name,
                                 line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=2.2)))
    if mark_x is not None:
        fig.add_vline(x=mark_x, line_dash="dash", line_color=NAVY, line_width=1.2,
                      annotation_text=f"dipilih: {mark_x}", annotation_font_size=10)
    fig.update_xaxes(title=x, dtick=1); fig.update_yaxes(title=ytitle)
    return style_fig(fig, height, len(ys) > 1, title)


def cv_bar(df: pd.DataFrame, value: str, sd: str, label: str, selected: str, title: str, height: int = 330) -> go.Figure:
    d = df.sort_values(value, ascending=True)
    fig = go.Figure(go.Bar(y=d[label], x=d[value], orientation="h",
                           marker_color=[CORAL if m == selected else TEAL for m in d[label]],
                           error_x=dict(type="data", array=d[sd], color=SLATE, thickness=1.2),
                           text=[f"{v:.3f}" for v in d[value]], textposition="inside", insidetextanchor="start",
                           textfont=dict(color="white", size=12),
                           hovertemplate="%{y}<br>%{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="Rata-rata cross-validation (garis = ±1 SD antar-fold)", range=[0, float((d[value] + d[sd]).max()) * 1.06])
    return style_fig(fig, height, False, title)
