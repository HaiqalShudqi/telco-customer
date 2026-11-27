"""Perhitungan analitik. Semua angka dihitung dari data aktif, tidak ada angka yang diketik manual."""
from __future__ import annotations

import pandas as pd

from utils.visualizations import seg_short

MIN_GROUP = 30  # kelompok lebih kecil dari ini tidak dipakai untuk membuat klaim


def has_churn(df: pd.DataFrame) -> bool:
    return "Churn_flag" in df.columns and df["Churn_flag"].notna().any()


def compute_kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    churn = has_churn(df)
    return {
        "n": n,
        "has_churn": churn,
        "n_churn": int(df["Churn_flag"].sum()) if churn else None,
        "churn_rate": float(df["Churn_flag"].mean() * 100) if churn and n else None,
        "tenure_mean": float(df["tenure"].mean()) if n else None,
        "monthly_mean": float(df["MonthlyCharges"].mean()) if n else None,
        "n_high_risk": int((df["risk_tier"] == "High").sum()),
        "pct_high_risk": float((df["risk_tier"] == "High").mean() * 100) if n else None,
    }


def churn_by(df: pd.DataFrame, col: str, min_n: int = MIN_GROUP) -> pd.DataFrame:
    """Churn rate per kategori (hanya kategori dengan n >= min_n)."""
    g = df.groupby(col, observed=True)["Churn_flag"].agg(n="count", churn_rate="mean").reset_index()
    g["churn_rate"] *= 100
    return g[g["n"] >= min_n].sort_values("churn_rate", ascending=False)


def executive_summary(df: pd.DataFrame, overall: pd.DataFrame) -> list[str]:
    """Poin ringkasan otomatis dari data yang sedang ditampilkan. Bersifat deskriptif (asosiasi)."""
    pts: list[str] = []
    n = len(df)
    if n == 0:
        return pts
    churn = has_churn(df)

    if churn:
        k = compute_kpis(df)
        base = overall["Churn_flag"].mean() * 100
        txt = f"Dari **{n:,}** pelanggan yang ditampilkan, **{k['n_churn']:,}** churn (**{k['churn_rate']:.1f}%**)."
        if n != len(overall):
            diff = k["churn_rate"] - base
            txt += f" Seluruh basis pelanggan: {base:.1f}% (selisih {diff:+.1f} poin persentase)."
        pts.append(txt)

        seg = churn_by(df, "segment_name")
        if len(seg) >= 2:
            top, low = seg.iloc[0], seg.iloc[-1]
            pts.append(f"Segmen dengan churn tertinggi: **{seg_short(top['segment_name'])}** "
                       f"({top['churn_rate']:.1f}%, n = {int(top['n']):,}); terendah: **{seg_short(low['segment_name'])}** "
                       f"({low['churn_rate']:.1f}%, n = {int(low['n']):,}).")

        con = churn_by(df, "Contract")
        if len(con) >= 2:
            top, low = con.iloc[0], con.iloc[-1]
            ratio = top["churn_rate"] / low["churn_rate"] if low["churn_rate"] > 0 else None
            extra = f", sekitar {ratio:.1f} kali lipat" if ratio else ""
            pts.append(f"Kontrak **{top['Contract']}** memiliki churn {top['churn_rate']:.1f}% dibanding "
                       f"{low['churn_rate']:.1f}% pada kontrak **{low['Contract']}**{extra}.")

    kh = int((df["risk_tier"] == "High").sum())
    if kh:
        rar_total = df["revenue_at_risk"].sum()
        share = df.loc[df["risk_tier"] == "High", "revenue_at_risk"].sum() / rar_total * 100 if rar_total else 0
        pts.append(f"Terdapat **{kh:,}** pelanggan berisiko tinggi ({kh / n * 100:.1f}%). Kelompok ini menyumbang "
                   f"**{share:.0f}%** dari indeks *revenue at risk* (proksi prioritisasi, bukan prakiraan kerugian).")

    seg_n = df["segment_name"].value_counts()
    if len(seg_n):
        pts.append(f"Segmen terbesar: **{seg_short(seg_n.index[0])}** dengan {seg_n.iloc[0]:,} pelanggan "
                   f"({seg_n.iloc[0] / n * 100:.1f}%).")
    return pts


# ==============================================================================================
# Stage 4: segmentasi, churn, dan tabel bantu
# ==============================================================================================
import numpy as np  # noqa: E402

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
ADDON_LABELS = {"OnlineSecurity": "Online Security", "OnlineBackup": "Online Backup",
                "DeviceProtection": "Device Protection", "TechSupport": "Tech Support",
                "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming Movies"}
TENURE_ORDER = ["0-12", "13-24", "25-48", "49-72"]


def wilson_ci(k: float, n: float, z: float = 1.959963984540054) -> tuple[float, float]:
    """Interval kepercayaan Wilson untuk proporsi (hasil dalam persen)."""
    if not n:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half) * 100, (centre + half) * 100


def segment_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Profil per segmen, dihitung dari data aktif (definisi kolom sama dengan segment_profiles.csv notebook)."""
    sid = df["segment_id"]
    g = df.groupby("segment_id")
    share = lambda cond: cond.groupby(sid).mean() * 100  # noqa: E731
    prof = pd.DataFrame({
        "segment_name": g["segment_name"].first(),
        "n": g.size(),
        "share_%": g.size() / len(df) * 100,
        "tenure_mean": g["tenure"].mean(),
        "monthly_mean": g["MonthlyCharges"].mean(),
        "total_mean": g["TotalCharges"].mean(),
        "n_addon_mean": g["n_addon_services"].mean(),
        "annual_value_mean": g["annual_value_proxy"].mean(),
        "pct_month_to_month": share(df["Contract"] == "Month-to-month"),
        "pct_one_year": share(df["Contract"] == "One year"),
        "pct_two_year": share(df["Contract"] == "Two year"),
        "pct_fiber": share(df["InternetService"] == "Fiber optic"),
        "pct_dsl": share(df["InternetService"] == "DSL"),
        "pct_no_internet": share(df["InternetService"] == "No"),
        "pct_echeck": share(df["PaymentMethod"] == "Electronic check"),
        "pct_paperless": share(df["PaperlessBilling"] == "Yes"),
        "pct_senior": share(df["SeniorCitizen"] == "Yes"),
        "pct_partner": share(df["Partner"] == "Yes"),
        "pct_dependents": share(df["Dependents"] == "Yes"),
        "prob_cal_mean_%": g["churn_prob_cal"].mean() * 100,
        "pct_high_risk": share(df["risk_tier"] == "High"),
        "revenue_at_risk_total": g["revenue_at_risk"].sum(),
    })
    if has_churn(df):
        k = g["Churn_flag"].sum()
        prof["churn_rate_%"] = g["Churn_flag"].mean() * 100
        ci = [wilson_ci(k[i], prof.loc[i, "n"]) for i in prof.index]
        prof["churn_ci_low_%"] = [c[0] for c in ci]
        prof["churn_ci_high_%"] = [c[1] for c in ci]
    else:
        prof["churn_rate_%"] = prof["churn_ci_low_%"] = prof["churn_ci_high_%"] = np.nan
    return prof


def overall_profile(df: pd.DataFrame) -> pd.Series:
    return segment_profile(df.assign(segment_id=0, segment_name="Semua")).iloc[0]


def service_adoption(df: pd.DataFrame) -> pd.DataFrame:
    """Adopsi layanan per segmen (%). Telepon dan internet: dari semua pelanggan segmen; add-on: dari pelanggan internet."""
    rows = {}
    for sid, s in df.groupby("segment_id"):
        inet = s[s["InternetService"] != "No"]
        r = {"Layanan telepon": (s["PhoneService"] == "Yes").mean() * 100,
             "Layanan internet": (s["InternetService"] != "No").mean() * 100}
        for a in ADDONS:
            r[ADDON_LABELS[a]] = (inet[a] == "Yes").mean() * 100 if len(inet) >= MIN_GROUP else np.nan
        rows[sid] = r
    return pd.DataFrame(rows).T


PCT_LABELS = {
    "pct_month_to_month": "kontrak bulanan", "pct_one_year": "kontrak 1 tahun", "pct_two_year": "kontrak 2 tahun",
    "pct_fiber": "internet fiber optic", "pct_dsl": "internet DSL", "pct_no_internet": "tanpa layanan internet",
    "pct_echeck": "membayar dengan electronic check", "pct_paperless": "paperless billing",
    "pct_senior": "senior citizen", "pct_partner": "memiliki partner", "pct_dependents": "memiliki tanggungan",
}


def _cmp(x: float, ref: float, rel: float = 0.15) -> str:
    if ref == 0 or np.isnan(ref):
        return "setara"
    d = (x - ref) / ref
    return "lebih tinggi" if d > rel else ("lebih rendah" if d < -rel else "setara")


def segment_narrative(row: pd.Series, ov: pd.Series) -> list[str]:
    """Interpretasi segmen dari angka aktual (aturan eksplisit, bukan teks karangan)."""
    out = [f"**Ukuran:** {int(row['n']):,} pelanggan ({row['share_%']:.1f}% dari basis yang ditampilkan)."]
    t = _cmp(row["tenure_mean"], ov["tenure_mean"])
    out.append(f"**Masa berlangganan:** rata-rata {row['tenure_mean']:.0f} bulan, {t} dibanding rata-rata keseluruhan "
               f"({ov['tenure_mean']:.0f} bulan).")
    m = _cmp(row["monthly_mean"], ov["monthly_mean"])
    out.append(f"**Tagihan:** ${row['monthly_mean']:.2f} per bulan ({m} dibanding ${ov['monthly_mean']:.2f}); "
               f"total tagihan rata-rata ${row['total_mean']:,.0f}.")
    cons = {"kontrak bulanan": row["pct_month_to_month"], "kontrak 1 tahun": row["pct_one_year"],
            "kontrak 2 tahun": row["pct_two_year"]}
    top_c = max(cons, key=cons.get)
    out.append(f"**Kontrak:** didominasi {top_c} ({cons[top_c]:.0f}%).")
    inets = {"internet fiber optic": row["pct_fiber"], "internet DSL": row["pct_dsl"], "tanpa internet": row["pct_no_internet"]}
    top_i = max(inets, key=inets.get)
    out.append(f"**Layanan:** terbanyak {top_i} ({inets[top_i]:.0f}%); rata-rata {row['n_addon_mean']:.1f} layanan tambahan "
               f"({_cmp(row['n_addon_mean'], ov['n_addon_mean'])} dibanding {ov['n_addon_mean']:.1f}).")

    pcts = [c for c in PCT_LABELS]
    dev = sorted(((abs(row[c] - ov[c]), c) for c in pcts), reverse=True)
    dist = [f"{row[c]:.0f}% {PCT_LABELS[c]} (keseluruhan {ov[c]:.0f}%)" for d, c in dev[:3] if d >= 10]
    if dist:
        out.append("**Ciri pembeda utama:** " + "; ".join(dist) + ".")

    if not np.isnan(row["churn_rate_%"]):
        ratio = row["churn_rate_%"] / ov["churn_rate_%"] if ov["churn_rate_%"] else np.nan
        out.append(f"**Churn:** {row['churn_rate_%']:.1f}% (CI 95% {row['churn_ci_low_%']:.1f}% sampai "
                   f"{row['churn_ci_high_%']:.1f}%), sekitar {ratio:.1f} kali rata-rata keseluruhan ({ov['churn_rate_%']:.1f}%).")
    out.append(f"**Risiko prediksi:** {row['pct_high_risk']:.1f}% pelanggan di segmen ini berada pada tingkat risiko tinggi.")
    return out


# ---------------------------------------------------------------- churn per kategori
def rate_table(df: pd.DataFrame, col: str, order: list | None = None, min_n: int = MIN_GROUP) -> pd.DataFrame:
    """Churn rate per kategori beserta CI 95% Wilson dan lift terhadap rata-rata."""
    base = df["Churn_flag"].mean() * 100
    g = df.groupby(col, observed=True)["Churn_flag"].agg(n="count", n_churn="sum").reset_index()
    g = g[g["n"] >= min_n].copy()
    g["churn_rate"] = g["n_churn"] / g["n"] * 100
    ci = [wilson_ci(k, n) for k, n in zip(g["n_churn"], g["n"])]
    g["ci_low"] = [c[0] for c in ci]
    g["ci_high"] = [c[1] for c in ci]
    g["lift"] = g["churn_rate"] / base
    if order:
        g[col] = pd.Categorical(g[col], categories=[o for o in order if o in set(g[col])], ordered=True)
        g = g.sort_values(col)
    else:
        g = g.sort_values("churn_rate", ascending=False)
    return g.reset_index(drop=True)


GAP_VARS = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
            "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group"] + ADDONS


def gap_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Peringkat variabel menurut selisih churn rate kategori tertinggi vs terendah (add-on: hanya pelanggan internet)."""
    rows = []
    for c in GAP_VARS:
        d = df[df["InternetService"] != "No"] if c in ADDONS else df
        t = rate_table(d, c)
        if len(t) < 2:
            continue
        hi, lo = t.iloc[0], t.iloc[-1]
        label = f"{c} (pelanggan internet)" if c in ADDONS else c
        rows.append({"variabel": label, "tertinggi": f"{hi[c]}", "churn_tertinggi": hi["churn_rate"],
                     "terendah": f"{lo[c]}", "churn_terendah": lo["churn_rate"], "gap_pp": hi["churn_rate"] - lo["churn_rate"]})
    return pd.DataFrame(rows).sort_values("gap_pp", ascending=False).reset_index(drop=True)


def smoothed_churn_by_tenure(df: pd.DataFrame, window: int = 5) -> pd.Series:
    by = df.groupby("tenure")["Churn_flag"].mean()
    return by.rolling(window, center=True, min_periods=1).mean() * 100


def addon_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    """Churn rate pelanggan internet menurut kepemilikan tiap layanan tambahan."""
    inet = df[df["InternetService"] != "No"]
    rows = []
    for a in ADDONS:
        for val in ("Yes", "No"):
            s = inet[inet[a] == val]
            if len(s) >= MIN_GROUP:
                rows.append({"Layanan": ADDON_LABELS[a], "Memiliki": "Ya" if val == "Yes" else "Tidak",
                             "n": len(s), "churn_rate": s["Churn_flag"].mean() * 100})
    return pd.DataFrame(rows)


def friendly_term(term: str) -> str:
    """'C(InternetService)[T.Fiber optic]' -> 'InternetService: Fiber optic'."""
    import re
    m = re.match(r"C\((\w+)\)\[T\.(.+)\]", term)
    return f"{m.group(1)}: {m.group(2)}" if m else term


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------------------------------------
# Ringkasan kualitas data (dihitung langsung dari dataset mentah)
# ----------------------------------------------------------------------------------------------
def data_quality(raw: pd.DataFrame) -> dict:
    """Pemeriksaan kualitas data pada dataset mentah, dengan aturan yang sama seperti Tahap 2 notebook."""
    n = len(raw)
    tc_txt = raw["TotalCharges"].astype(str).str.strip()
    blank = tc_txt.eq("")
    tc = pd.to_numeric(tc_txt.where(~blank), errors="coerce")
    out = {
        "n_rows": n, "n_cols": raw.shape[1], "tc_dtype": str(raw["TotalCharges"].dtype),
        "n_blank_tc": int(blank.sum()),
        "blank_all_tenure0": bool((raw.loc[blank, "tenure"] == 0).all()) if blank.any() else None,
        "blank_all_no_churn": bool((raw.loc[blank, "Churn"] == "No").all()) if blank.any() and "Churn" in raw else None,
        "n_tenure0": int((raw["tenure"] == 0).sum()),
        "n_full_dup": int(raw.duplicated().sum()), "n_id_dup": int(raw["customerID"].duplicated().sum()),
        "n_profile_dup": int(raw.drop(columns="customerID").duplicated().sum()),
    }
    out_n = {}
    for c, s in (("tenure", raw["tenure"]), ("MonthlyCharges", raw["MonthlyCharges"]), ("TotalCharges", tc)):
        s = s.dropna(); q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
        out_n[c] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    out["iqr_outliers"] = out_n
    inet_no = raw["InternetService"] == "No"
    viol = (inet_no & (raw[ADDONS] != "No internet service").any(axis=1)) | (~inet_no & (raw[ADDONS] == "No internet service").any(axis=1))
    viol |= ((raw["PhoneService"] == "No") & (raw["MultipleLines"] != "No phone service")) | \
            ((raw["PhoneService"] == "Yes") & (raw["MultipleLines"] == "No phone service"))
    out["n_logic_violations"] = int(viol.sum())
    exp = raw["tenure"] * raw["MonthlyCharges"]
    ok = tc.notna()
    out["corr_total_vs_expected"] = float(np.corrcoef(tc[ok], exp[ok])[0, 1])
    if "Churn" in raw:
        vc = raw["Churn"].value_counts()
        out["churn_pct"] = float(vc.get("Yes", 0) / n * 100)
        out["class_ratio"] = float(vc.max() / vc.min())
    return out
