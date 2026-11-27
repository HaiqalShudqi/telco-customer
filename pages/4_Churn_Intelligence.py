"""Halaman 4: Churn Intelligence (eksplorasi churn + hasil uji statistik dari notebook)."""
import numpy as np
import pandas as pd
import streamlit as st

from utils import config, ui
from utils.analytics import (ADDON_LABELS, ADDONS, TENURE_ORDER, addon_yes_no, friendly_term, fmt_p, gap_ranking, has_churn,
                             rate_table, smoothed_churn_by_tenure)
from utils.data_loader import load_report_csv, load_report_json
from utils.visualizations import (CHURN_COLORS, CORAL, SLATE, TEAL, forest, grouped_bar, hbar, heatmap, histogram, line_chart,
                                  rate_bar, donut)

ui.bootstrap()
table, source_label, info = ui.get_dataset()
ui.page_header("Churn Intelligence",
               "Di mana churn terkonsentrasi, dan hasil uji statistik dari notebook beserta artinya dalam bahasa sederhana.")
st.caption(f"Sumber data: {source_label}")

churn = has_churn(table)
YN = {"Yes": "Ya", "No": "Tidak"}
overall = float(table["Churn_flag"].mean() * 100) if churn else float("nan")


def rate_chart(col, title, df=None, order=None, labels=None, height=300, key=None):
    d = table if df is None else df
    rt = rate_table(d, col, order)
    ui.plot(rate_bar(rt, col, overall, title=title, height=height, labels=labels), key=key or f"rc_{col}_{title}")
    return rt


def finding(rt, col, name):
    """Kalimat temuan dari tabel rate_table (tertinggi vs terendah)."""
    if len(rt) < 2:
        return None
    r = rt.sort_values("churn_rate", ascending=False)
    hi, lo = r.iloc[0], r.iloc[-1]
    return (f"**{name}:** churn tertinggi pada **{hi[col]}** ({hi['churn_rate']:.1f}%), terendah pada **{lo[col]}** "
            f"({lo['churn_rate']:.1f}%), selisih {hi['churn_rate'] - lo['churn_rate']:.1f} poin persentase.")


def findings_box(lines):
    with st.container(border=True):
        for ln in [x for x in lines if x]:
            st.markdown(f"- {ln}")


tabs = st.tabs(["Ringkasan", "Demografi", "Tenure", "Kontrak & pembayaran", "Layanan", "Hasil statistik", "Faktor risiko & survival"])

NO_CHURN_MSG = ("Dataset aktif tidak memiliki kolom Churn, sehingga analisis churn aktual tidak tersedia di tab ini. "
                "Tab *Hasil statistik* dan *Faktor risiko & survival* tetap menampilkan hasil notebook pada dataset bawaan.")

# ================================================================== Ringkasan
with tabs[0]:
    if not churn:
        ui.note(NO_CHURN_MSG, warn=True)
    else:
        n_ch = int(table["Churn_flag"].sum())
        ch, ret = table[table["Churn_flag"] == 1], table[table["Churn_flag"] == 0]
        ui.kpi_row([
            ("Churn rate", f"{overall:.1f}%", f"{n_ch:,} dari {len(table):,} pelanggan", "#D1495B"),
            ("Pelanggan bertahan", f"{len(ret):,}", f"{100 - overall:.1f}%", "#0F6E8C"),
            ("Median tenure: churn", f"{ch['tenure'].median():.0f} bln", f"non-churn: {ret['tenure'].median():.0f} bln", "#6C5B9E"),
            ("Median tagihan: churn", f"${ch['MonthlyCharges'].median():.2f}", f"non-churn: ${ret['MonthlyCharges'].median():.2f}", "#F2A541"),
        ])
        c1, c2 = st.columns([1, 1.6])
        with c1:
            d = table.assign(status=table["Churn_flag"].map({1: "Yes", 0: "No"}))
            ui.plot(donut(d, "status", CHURN_COLORS, order=["No", "Yes"], title="Distribusi churn",
                          hole_text=f"{overall:.1f}%<br>churn"), key="ci_donut")
        gap = gap_ranking(table)
        with c2:
            top = gap.head(10)
            ui.plot(hbar(top["variabel"].tolist(), top["gap_pp"].tolist(), CORAL, height=330, fmt="{:.1f} pp",
                         title="10 variabel dengan selisih churn terbesar antar-kategori", xtitle="Selisih churn rate (poin persentase)"),
                    key="ci_gap")
        ui.section("Temuan utama")
        findings_box([f"**{r['variabel']}**: {r['tertinggi']} ({r['churn_tertinggi']:.1f}%) vs {r['terendah']} "
                      f"({r['churn_terendah']:.1f}%)" for _, r in gap.head(5).iterrows()])
        st.caption("Selisih besar belum berarti sebab-akibat, dan beberapa variabel saling berkaitan (misalnya kontrak dan tenure). "
                   "Layanan tambahan dihitung hanya pada pelanggan internet.")
        with st.expander("Tabel lengkap peringkat variabel"):
            ui.dataframe(gap.rename(columns={"tertinggi": "kategori churn tertinggi", "churn_tertinggi": "churn tertinggi (%)",
                                             "terendah": "kategori churn terendah", "churn_terendah": "churn terendah (%)",
                                             "gap_pp": "selisih (pp)"}).round(2), hide_index=True)

# ================================================================== Demografi
with tabs[1]:
    if not churn:
        ui.note(NO_CHURN_MSG, warn=True)
    else:
        a, b = st.columns(2)
        with a:
            r_g = rate_chart("gender", "Churn rate menurut gender")
        with b:
            r_s = rate_chart("SeniorCitizen", "Churn rate menurut senior citizen", labels=YN)
        c, d_ = st.columns(2)
        with c:
            r_p = rate_chart("Partner", "Churn rate menurut partner", labels=YN)
        with d_:
            r_d = rate_chart("Dependents", "Churn rate menurut tanggungan", labels=YN)
        combo = table.assign(**{"Partner & tanggungan": np.select(
            [(table["Partner"] == "Yes") & (table["Dependents"] == "Yes"), (table["Partner"] == "Yes"),
             (table["Dependents"] == "Yes")], ["Partner + tanggungan", "Partner saja", "Tanggungan saja"], "Tidak keduanya")})
        r_c = rate_chart("Partner & tanggungan", "Churn rate menurut kombinasi partner dan tanggungan", df=combo, height=320,
                         key="ci_combo")
        ui.section("Temuan")
        findings_box([finding(r_g, "gender", "Gender"), finding(r_s, "SeniorCitizen", "Senior citizen"),
                      finding(r_p, "Partner", "Partner"), finding(r_d, "Dependents", "Tanggungan"),
                      finding(r_c, "Partner & tanggungan", "Kombinasi")])
        st.caption("Batang merah: di atas rata-rata keseluruhan; biru: di bawahnya. Garis kecil pada batang adalah CI 95%. "
                   "Hasil uji signifikansi tersedia di tab *Hasil statistik*.")

# ================================================================== Tenure
with tabs[2]:
    if not churn:
        ui.note(NO_CHURN_MSG, warn=True)
    else:
        a, b = st.columns(2)
        with a:
            r_t = rate_chart("tenure_group", "Churn rate menurut kelompok tenure (bulan)", order=TENURE_ORDER, key="ci_tg")
        with b:
            ui.plot(line_chart(smoothed_churn_by_tenure(table), overall, "Churn rate menurut tenure (rata-rata bergerak 5 bulan)",
                               "Tenure (bulan)", "Churn rate (%)"), key="ci_tline")
        lab = table.assign(Churn=table["Churn_flag"].map({1: "Yes", 0: "No"}))
        ui.plot(histogram(lab, "tenure", "Churn", CHURN_COLORS, nbins=36, xtitle="Tenure (bulan)",
                          title="Distribusi tenure menurut status churn", height=300), key="ci_thist")
        ui.section("Temuan")
        ch_med = table.loc[table["Churn_flag"] == 1, "tenure"].median()
        nc_med = table.loc[table["Churn_flag"] == 0, "tenure"].median()
        findings_box([finding(r_t, "tenure_group", "Kelompok tenure"),
                      f"Median tenure pelanggan churn **{ch_med:.0f} bulan** vs **{nc_med:.0f} bulan** pada pelanggan yang bertahan."])
        st.caption("Pola ini konsisten dengan churn yang terkonsentrasi di masa awal berlangganan.")

# ================================================================== Kontrak & pembayaran
with tabs[3]:
    if not churn:
        ui.note(NO_CHURN_MSG, warn=True)
    else:
        a, b = st.columns(2)
        with a:
            r_ct = rate_chart("Contract", "Churn rate menurut jenis kontrak", order=["Month-to-month", "One year", "Two year"], key="ci_ct")
        with b:
            r_pm = rate_chart("PaymentMethod", "Churn rate menurut metode pembayaran", key="ci_pm")
        a, b = st.columns([1, 1.6])
        with a:
            r_pl = rate_chart("PaperlessBilling", "Churn rate menurut paperless billing", labels=YN, key="ci_pl")
        with b:
            g = table.groupby(["Contract", "PaymentMethod"])["Churn_flag"].agg(["mean", "count"]).reset_index()
            g.loc[g["count"] < 30, "mean"] = np.nan
            z = g.pivot(index="Contract", columns="PaymentMethod", values="mean").reindex(["Month-to-month", "One year", "Two year"]) * 100
            ui.plot(heatmap(z, colorscale="Reds", fmt="{:.1f}%", height=300, title="Churn rate (%): kontrak x metode pembayaran",
                            colorbar_title="%"), key="ci_cp")
        ui.section("Temuan")
        findings_box([finding(r_ct, "Contract", "Kontrak"), finding(r_pm, "PaymentMethod", "Metode pembayaran"),
                      finding(r_pl, "PaperlessBilling", "Paperless billing")])
        st.caption("Sel heatmap dengan kurang dari 30 pelanggan dikosongkan karena estimasinya tidak stabil.")

# ================================================================== Layanan
with tabs[4]:
    if not churn:
        ui.note(NO_CHURN_MSG, warn=True)
    else:
        a, b, c = st.columns(3)
        with a:
            r_i = rate_chart("InternetService", "Churn rate menurut layanan internet", key="ci_is")
        with b:
            r_ph = rate_chart("PhoneService", "Churn rate menurut layanan telepon", labels=YN, key="ci_ph")
        with c:
            r_ml = rate_chart("MultipleLines", "Churn rate menurut multiple lines", labels=YN, key="ci_ml")
        inet = table[table["InternetService"] != "No"]
        ay = addon_yes_no(table)
        ui.plot(grouped_bar(ay, "Layanan", "churn_rate", "Memiliki", {"Ya": TEAL, "Tidak": CORAL},
                            title="Churn rate menurut kepemilikan layanan tambahan (hanya pelanggan internet)", height=340), key="ci_addon")
        a, b = st.columns(2)
        with a:
            ui.plot(rate_bar(rate_table(inet, "n_addon_services"), "n_addon_services", float(inet["Churn_flag"].mean() * 100),
                             title="Churn rate menurut jumlah layanan tambahan (pelanggan internet)"), key="ci_naddon")
        ui.section("Temuan")
        lines = [finding(r_i, "InternetService", "Layanan internet")]
        for a_ in ADDONS:
            y = ay[(ay["Layanan"] == ADDON_LABELS[a_]) & (ay["Memiliki"] == "Ya")]
            n_ = ay[(ay["Layanan"] == ADDON_LABELS[a_]) & (ay["Memiliki"] == "Tidak")]
            if len(y) and len(n_):
                lines.append(f"**{ADDON_LABELS[a_]}:** churn {y['churn_rate'].iloc[0]:.1f}% pada pelanggan yang memilikinya vs "
                             f"{n_['churn_rate'].iloc[0]:.1f}% pada yang tidak.")
        findings_box(lines)
        st.caption("Perbandingan hanya pada pelanggan internet karena pelanggan tanpa internet tidak dapat memiliki layanan tambahan. "
                   "Streaming bukan layanan perlindungan sehingga polanya dapat berbeda dari Security/Support.")

# ================================================================== Hasil statistik
SIG_NOTE = ("Hasil di tab ini dihitung di notebook pada <b>dataset IBM bawaan</b> (7.043 pelanggan) dan tidak berubah ketika Anda "
            "mengunggah dataset lain. Semua hubungan adalah asosiasi, bukan sebab-akibat.")


def eps_label(e):
    return "diabaikan" if e < 0.01 else ("kecil" if e < 0.08 else ("sedang" if e < 0.26 else "besar"))


with tabs[5]:
    ui.note(SIG_NOTE)
    chi = load_report_csv("metrics/stat_chi_square.csv")
    mw = load_report_csv("metrics/stat_mann_whitney.csv")
    kw = load_report_csv("metrics/stat_kruskal_wallis.csv")
    if chi is None and mw is None and kw is None:
        st.warning("File hasil uji statistik tidak ditemukan di folder reports/metrics.")
    sub = st.tabs(["Chi-square & Cramér's V", "Mann-Whitney U", "Kruskal-Wallis"])

    with sub[0]:
        if chi is None:
            st.warning("File stat_chi_square.csv tidak ditemukan.")
        else:
            chi = chi.sort_values("cramers_v", ascending=False)
            st.markdown("**Apa yang diuji?** Apakah proporsi churn berbeda antar kategori suatu variabel (H0: tidak ada hubungan). "
                        "P-value dikoreksi Holm karena banyak variabel diuji sekaligus. **Cramér's V** mengukur *kekuatan* hubungan "
                        "(0 = tidak ada, 1 = sempurna): < 0,1 sangat lemah, 0,1-0,2 lemah, 0,2-0,3 sedang, ≥ 0,3 kuat.")
            ui.plot(hbar(chi["variabel"].tolist(), chi["cramers_v"].tolist(),
                         [CORAL if s else SLATE for s in chi["signifikan_holm"]], height=480, fmt="{:.3f}",
                         title="Kekuatan asosiasi dengan churn (Cramér's V); abu-abu = tidak signifikan", xtitle="Cramér's V"),
                    key="ci_cramer")
            show = chi[["variabel", "chi2", "dof", "p_holm", "cramers_v", "kekuatan_asosiasi", "signifikan_holm"]].copy()
            show["p_holm"] = show["p_holm"].map(fmt_p)
            show["signifikan_holm"] = show["signifikan_holm"].map({True: "Ya", False: "Tidak"})
            show.columns = ["Variabel", "Chi-square", "df", "p (Holm)", "Cramér's V", "Kekuatan", "Signifikan"]
            ui.dataframe(show.round({"Chi-square": 1, "Cramér's V": 3}), hide_index=True)
            sig, nsig = chi[chi["signifikan_holm"]], chi[~chi["signifikan_holm"]]
            top3 = ", ".join(f"{r['variabel']} (V = {r['cramers_v']:.3f})" for _, r in chi.head(3).iterrows())
            st.markdown("**Artinya:**")
            findings_box([
                f"**{len(sig)} dari {len(chi)}** variabel berasosiasi signifikan dengan churn setelah koreksi Holm.",
                f"Hubungan terkuat: {top3}. Variabel-variabel ini paling membedakan pelanggan churn dan bertahan.",
                ("Tidak ditemukan hubungan signifikan untuk: " + ", ".join(nsig["variabel"]) + ".") if len(nsig) else None,
                "P-value yang sangat kecil terjadi hampir pada semua variabel karena sampel besar; gunakan Cramér's V untuk menilai "
                "seberapa penting hubungannya.",
            ])

    with sub[1]:
        if mw is None:
            st.warning("File stat_mann_whitney.csv tidak ditemukan.")
        else:
            st.markdown("**Apa yang diuji?** Apakah distribusi variabel numerik berbeda antara pelanggan churn dan non-churn. Dipilih uji "
                        "nonparametrik karena ketiga variabel tidak berdistribusi normal. **Rank-biserial r** adalah ukuran efek "
                        "(-1 sampai 1; tanda negatif = pelanggan churn cenderung bernilai lebih rendah): < 0,1 diabaikan, 0,1-0,3 kecil, "
                        "0,3-0,5 sedang, > 0,5 besar. Welch t-test dan Cohen's d hanya sebagai pembanding.")
            show = mw[["variabel", "median_churn", "median_non_churn", "rank_biserial_r", "besar_efek", "cohens_d", "p_holm"]].copy()
            show["p_holm"] = show["p_holm"].map(fmt_p)
            show.columns = ["Variabel", "Median churn", "Median non-churn", "Rank-biserial r", "Besar efek", "Cohen's d", "p (Holm)"]
            ui.dataframe(show.round(3), hide_index=True)
            st.markdown("**Artinya:**")
            lines = []
            for _, r in mw.iterrows():
                arah = "lebih rendah" if r["rank_biserial_r"] < 0 else "lebih tinggi"
                lines.append(f"**{r['variabel']}:** median pelanggan churn {r['median_churn']:,.2f} vs {r['median_non_churn']:,.2f} pada "
                             f"non-churn. Pelanggan churn cenderung bernilai {arah} (r = {r['rank_biserial_r']:.3f}, efek {r['besar_efek']}).")
            findings_box(lines)

    with sub[2]:
        if kw is None:
            st.warning("File stat_kruskal_wallis.csv tidak ditemukan.")
        else:
            st.markdown("**Apa yang diuji?** Apakah distribusi variabel numerik berbeda di antara tiga kelompok atau lebih (pengganti ANOVA "
                        "untuk data non-normal). **Epsilon kuadrat (ε²)** adalah ukuran efek; patokan yang dipakai di sini: < 0,01 diabaikan, "
                        "< 0,08 kecil, < 0,26 sedang, ≥ 0,26 besar.")
            show = kw.copy()
            show["Besar efek"] = show["epsilon_sq"].map(eps_label)
            show["p_holm"] = show["p_holm"].map(fmt_p)
            show = show[["variabel", "kelompok", "H", "epsilon_sq", "Besar efek", "p_holm"]]
            show.columns = ["Variabel", "Kelompok", "H", "ε²", "Besar efek", "p (Holm)"]
            ui.dataframe(show.round({"H": 1, "ε²": 3}), hide_index=True)
            st.markdown("**Artinya:**")
            lines = []
            for _, r in kw.sort_values("epsilon_sq", ascending=False).iterrows():
                e = eps_label(r["epsilon_sq"])
                tail = "perbedaannya signifikan secara statistik tetapi sangat kecil secara praktis." if e == "diabaikan" else \
                       f"perbedaan antar kelompok {e}."
                lines.append(f"**{r['variabel']} menurut {r['kelompok']}:** ε² = {r['epsilon_sq']:.3f}; {tail}")
            findings_box(lines)
            st.caption("Uji post-hoc antar pasangan kontrak dijalankan di notebook (Holm), namun hasilnya tidak diekspor sebagai tabel sehingga tidak ditampilkan di sini.")

# ================================================================== Faktor risiko & survival
with tabs[6]:
    ui.note(SIG_NOTE)
    rr = load_report_csv("metrics/churn_relative_risk.csv")
    orr = load_report_csv("metrics/logit_odds_ratios.csv", index_col=0)
    surv = load_report_json("metrics/survival_summary.json")

    ui.section("Risiko relatif", "Berapa kali lebih sering pelanggan dengan karakteristik tertentu churn dibanding yang tidak.")
    if rr is None:
        st.warning("File churn_relative_risk.csv tidak ditemukan.")
    else:
        rr = rr.sort_values("risiko_relatif", ascending=False)
        ui.plot(hbar(rr["karakteristik"].tolist(), rr["risiko_relatif"].tolist(), CORAL, height=420, fmt="{:.2f}x",
                     title="Risiko relatif churn (1 = sama dengan pelanggan lain)", xtitle="Risiko relatif"), key="ci_rr")
        show = rr[["karakteristik", "n_ya", "churn_rate_ya_%", "churn_rate_tidak_%", "risiko_relatif", "p_holm"]].copy()
        show["p_holm"] = show["p_holm"].map(fmt_p)
        show.columns = ["Karakteristik", "Jumlah pelanggan", "Churn jika ya (%)", "Churn jika tidak (%)", "Risiko relatif", "p (Holm)"]
        ui.dataframe(show.round(2), hide_index=True)
        t = rr.iloc[0]
        st.markdown(f"**Artinya:** pelanggan dengan *{t['karakteristik'].lower()}* churn **{t['churn_rate_ya_%']:.1f}%** dibanding "
                    f"**{t['churn_rate_tidak_%']:.1f}%** pada pelanggan lain (sekitar {t['risiko_relatif']:.1f} kali lipat). "
                    "Karakteristik ini saling berkaitan, sehingga risiko relatifnya tidak boleh dijumlahkan.")

    ui.section("Regresi logistik terkontrol (odds ratio)",
               "Hubungan tiap faktor dengan churn setelah faktor lain dalam model dikontrol. OR > 1 menaikkan odds churn, OR < 1 menurunkannya.")
    if orr is None:
        st.warning("File logit_odds_ratios.csv tidak ditemukan.")
    else:
        o = orr.copy()
        o["label"] = [friendly_term(i) for i in o.index]
        o["sig"] = o["p_holm"] < 0.05
        o = o.sort_values("odds_ratio", ascending=False)
        ui.plot(forest(o, title="Odds ratio dan CI 95% (merah = signifikan setelah koreksi Holm)", height=60 + 26 * len(o)), key="ci_forest")
        show = o[["label", "odds_ratio", "ci_low", "ci_high", "p_holm", "sig"]].copy()
        show["p_holm"] = show["p_holm"].map(fmt_p)
        show["sig"] = show["sig"].map({True: "Ya", False: "Tidak"})
        show.columns = ["Faktor (level)", "Odds ratio", "CI 95% bawah", "CI 95% atas", "p (Holm)", "Signifikan"]
        with st.expander("Tabel odds ratio"):
            ui.dataframe(show.round(3), hide_index=True)
        ui.note("Level acuan mengikuti urutan alfabet: Contract = Month-to-month, InternetService = DSL, PaymentMethod = Bank transfer. "
                "<b>Tenure</b> dihitung per bulan dan <b>MonthlyCharges</b> per dolar. Nilai OR MonthlyCharges di bawah 1 kemungkinan "
                "terkait korelasi tagihan dengan layanan yang sudah ada di model, sehingga tidak boleh dibaca sebagai 'tagihan lebih tinggi "
                "mengurangi churn'. Layanan tambahan bernilai 'No' juga mencakup pelanggan tanpa internet.", warn=True)

    ui.section("Analisis survival (Kaplan-Meier)",
               "Probabilitas pelanggan masih berlangganan pada bulan tertentu. Pelanggan yang belum churn dianggap tersensor pada tenure saat ini (asumsi, karena data hanya satu snapshot).")
    if surv is None:
        st.warning("File survival_summary.json tidak ditemukan.")
    else:
        sc = pd.DataFrame(surv["survival_by_contract"]).T
        sc.columns = ["Bertahan sampai bulan 12", "Bertahan sampai bulan 24", "Bertahan sampai bulan 48"]
        a, b = st.columns([1.2, 1])
        with a:
            ui.dataframe((sc * 100).round(1).astype(str) + "%")
            st.caption("Probabilitas bertahan per jenis kontrak (dari output notebook).")
        with b:
            lr = pd.DataFrame(surv["logrank"]).T
            lr.index = ["Kontrak" if i == "Contract" else ("Layanan internet" if i == "InternetService" else "Segmen") for i in lr.index]
            lr["p_value"] = lr["p_value"].map(fmt_p)
            lr.columns = ["Chi-square", "p-value"]
            ui.dataframe(lr)
            st.caption("Uji log-rank: p < 0,001 berarti kurva bertahan berbeda antar kelompok.")
        m2m, tw = sc.loc["Month-to-month"], sc.loc["Two year"]
        st.markdown(f"**Artinya:** pada kontrak bulanan hanya **{m2m.iloc[0] * 100:.0f}%** pelanggan masih bertahan sampai bulan ke-12 "
                    f"dan **{m2m.iloc[2] * 100:.0f}%** sampai bulan ke-48, sedangkan pada kontrak dua tahun **{tw.iloc[0] * 100:.1f}%** dan "
                    f"**{tw.iloc[2] * 100:.1f}%**. Kontrak berjangka juga membatasi kapan pelanggan *boleh* berhenti, sehingga "
                    "perbedaan ini tidak otomatis berarti kontrak membuat pelanggan lebih puas.")
        img = config.REPORTS / "figures" / "07_kaplan_meier.png"
        if img.exists():
            st.image(str(img), caption="Kurva Kaplan-Meier menurut kontrak, layanan internet, dan segmen (dari notebook).")
