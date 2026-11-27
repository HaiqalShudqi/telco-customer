"""Halaman 7: Methodology & Model Performance. Semua angka berasal dari artefak notebook di folder reports/."""
import re

import numpy as np
import pandas as pd
import streamlit as st

from utils import config, ui
from utils.analytics import data_quality
from utils.data_loader import (DataLoadError, load_raw_default, load_report_csv, load_report_json, load_test_predictions)
from utils.evaluation import metrics_at, pr_points, roc_points, roc_pr_summary, threshold_table
from utils.visualizations import (CORAL, SLATE, TEAL, style_fig, confusion_figure, cv_bar, hbar, multi_line, pr_figure, roc_figure,
                                  threshold_figure)

import plotly.graph_objects as go

ui.bootstrap()
ui.get_dataset()  # memasang sidebar; halaman ini selalu memakai artefak notebook, bukan dataset aktif
ui.page_header("Methodology & Model Performance",
               "Bagaimana sistem ini dibangun: data, segmentasi, prediksi churn, evaluasi, interpretabilitas, dan keterbatasannya.")

R = config.REPORTS
ext = load_report_json("metrics/notebook_extracts.json") or {}
summ = load_report_json("metrics/metrics_summary.json") or {}
cvdf = load_report_csv("metrics/cv_model_comparison.csv")
testdf = load_report_csv("metrics/test_metrics_all_models.csv", index_col=0)
kms = load_report_csv("segmentation/kmeans_k_selection.csv")
mcmp = load_report_csv("segmentation/method_comparison.csv")
K = summ.get("k_final")
BEST = summ.get("best_model")
THR = summ.get("threshold")


@st.cache_data
def _dictionary():
    return pd.read_csv(config.DICT_FILE) if config.DICT_FILE.exists() else None


def missing(name: str) -> None:
    st.warning(f"Artefak {name} tidak ditemukan, bagian ini tidak dapat ditampilkan.")


def pm(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


tabs = st.tabs(["A. Data", "B. Segmentasi", "C. Prediksi churn", "D. Evaluasi model", "E. Interpretabilitas", "F. Keterbatasan"])

# ====================================================================================== A. DATA
with tabs[0]:
    try:
        raw = load_raw_default()
        dq = data_quality(raw)
    except DataLoadError as exc:
        st.error(str(exc)); dq = None
    if dq:
        ui.section("Deskripsi dataset")
        ui.kpi_row([("Observasi", f"{dq['n_rows']:,}", "satu baris per pelanggan", "#0F6E8C"),
                    ("Variabel", f"{dq['n_cols']}", "1 ID, 1 target, 19 prediktor", "#6C5B9E"),
                    ("Numerik", "3", "tenure, MonthlyCharges, TotalCharges", "#F2A541"),
                    ("Churn rate", f"{dq['churn_pct']:.2f}%", f"rasio kelas {dq['class_ratio']:.2f} : 1", "#D1495B")])
        st.write("")
        st.markdown("Dataset yang dipakai adalah **IBM Telco Customer Churn**: data pelanggan sebuah operator telekomunikasi yang mencakup "
                    "demografi, informasi akun dan kontrak, layanan yang dipakai, tagihan, dan status churn (apakah pelanggan berhenti). "
                    "Data berupa satu *snapshot* tanpa tanggal.")
        dd = _dictionary()
        if dd is not None:
            with st.expander("Kamus data (variabel, tipe, dan catatan kualitas)"):
                ui.dataframe(dd.rename(columns={"variabel": "Variabel", "kategori": "Kelompok", "deskripsi": "Deskripsi",
                                                "tipe_asli": "Tipe", "n_unique": "Nilai unik", "catatan_kualitas": "Catatan kualitas"}),
                             hide_index=True)
        ui.section("Temuan kualitas data", "Dihitung langsung dari dataset mentah dengan aturan yang sama seperti Tahap 2 notebook.")
        o = dq["iqr_outliers"]
        rows = [
            ("Tipe data TotalCharges", f"Bertipe `{dq['tc_dtype']}` (teks) padahal isinya angka.", "Dikonversi ke numerik."),
            ("Missing value tersembunyi",
             f"{dq['n_blank_tc']} nilai TotalCharges berisi spasi ({dq['n_blank_tc'] / dq['n_rows'] * 100:.2f}%). "
             f"Semuanya berstatus tenure = 0{' dan Churn = No' if dq['blank_all_no_churn'] else ''}"
             f"{' (sama dengan seluruh ' + str(dq['n_tenure0']) + ' pelanggan tenure 0)' if dq['blank_all_tenure0'] and dq['n_blank_tc'] == dq['n_tenure0'] else ''}.",
             "Diisi 0 (diinterpretasikan belum pernah ditagih); baris dipertahankan."),
            ("Duplikasi", f"Baris duplikat penuh: {dq['n_full_dup']}; customerID ganda: {dq['n_id_dup']}; baris identik selain ID: "
                          f"{dq['n_profile_dup']} ({dq['n_profile_dup'] / dq['n_rows'] * 100:.2f}%).",
             "Tidak ditindak (kemungkinan pelanggan berbeda dengan profil sama)."),
            ("Outlier (aturan IQR)", f"tenure: {o['tenure']}, MonthlyCharges: {o['MonthlyCharges']}, TotalCharges: {o['TotalCharges']}.",
             "Tidak ada tindakan."),
            ("Konsistensi logika layanan", f"{dq['n_logic_violations']} pelanggaran (mis. tanpa internet tetapi punya add-on).", "Tidak diperlukan."),
            ("Redundansi nilai finansial",
             f"Korelasi TotalCharges dengan tenure × MonthlyCharges = {dq['corr_total_vs_expected']:.4f}.",
             "Dipertimbangkan saat memilih fitur clustering (TotalCharges tidak dipakai) dan model."),
            ("Kategori redundan", "'No internet service' dan 'No phone service' hanya mengulang informasi kolom induknya.",
             "Dinormalisasi menjadi 'No' pada feature engineering."),
            ("Ketidakseimbangan target", f"Churn {dq['churn_pct']:.2f}%, rasio kelas {dq['class_ratio']:.2f} : 1 (sedang).",
             "Class weighting; evaluasi dengan PR-AUC dan recall, bukan hanya accuracy."),
        ]
        ui.dataframe(pd.DataFrame(rows, columns=["Pemeriksaan", "Hasil", "Penanganan"]), hide_index=True)
        st.caption("Secara keseluruhan kualitas data baik; satu-satunya masalah nyata adalah TotalCharges.")

# ====================================================================================== B. SEGMENTASI
with tabs[1]:
    if kms is None or mcmp is None or not K:
        missing("kmeans_k_selection.csv / method_comparison.csv / metrics_summary.json")
    else:
        row = kms[kms["k"] == K].iloc[0]
        ui.section("Metodologi K-Means")
        st.markdown(
            "- **Fitur:** `tenure`, `MonthlyCharges`, `n_addon_services`, `contract_ord`, distandardisasi (StandardScaler). "
            "`TotalCharges` tidak dipakai karena redundan dengan tenure × MonthlyCharges.\n"
            "- **Churn tidak dipakai** membentuk cluster; hanya dipakai setelahnya untuk menilai kegunaan bisnis.\n"
            f"- **Aturan memilih k:** k antara 3 dan 6, cluster terkecil minimal 5%, lalu Silhouette tertinggi. k = 2 tetap dievaluasi "
            "tetapi jarang berguna sebagai segmentasi bisnis.\n"
            f"- **Hasil:** k = **{K}**.")
        ui.kpi_row([("Jumlah cluster", f"{K}", "K-Means", "#0F6E8C"),
                    ("Silhouette", f"{row['silhouette']:.4f}", "lebih tinggi lebih baik", "#2E933C"),
                    ("Davies-Bouldin", f"{row['davies_bouldin']:.4f}", "lebih rendah lebih baik", "#F2A541"),
                    ("Calinski-Harabasz", f"{row['calinski_harabasz']:,.0f}", "lebih tinggi lebih baik", "#6C5B9E"),
                    ("Cluster terkecil", f"{row['min_cluster_share'] * 100:.1f}%", "dari seluruh pelanggan", "#14213D")])
        st.write("")
        a, b = st.columns(2)
        with a:
            ui.plot(multi_line(kms, "k", [("silhouette", "Silhouette")], mark_x=K, title="Silhouette menurut k"), key="m_sil")
            ui.plot(multi_line(kms, "k", [("inertia", "Inertia")], mark_x=K, title="Elbow (inertia) menurut k"), key="m_inertia")
        with b:
            ui.plot(multi_line(kms, "k", [("davies_bouldin", "Davies-Bouldin")], mark_x=K, title="Davies-Bouldin menurut k"), key="m_db")
            ui.plot(multi_line(kms, "k", [("calinski_harabasz", "Calinski-Harabasz")], mark_x=K, title="Calinski-Harabasz menurut k"), key="m_ch")
        best_k = {"Silhouette": int(kms.loc[kms["silhouette"].idxmax(), "k"]),
                  "Davies-Bouldin": int(kms.loc[kms["davies_bouldin"].idxmin(), "k"]),
                  "Calinski-Harabasz": int(kms.loc[kms["calinski_harabasz"].idxmax(), "k"])}
        agree = all(v == K for v in best_k.values())
        st.markdown("**k terbaik menurut tiap indeks:** " + ", ".join(f"{n} → k = {v}" for n, v in best_k.items()) + ". "
                    + ("Ketiga indeks sepakat pada k yang dipilih." if agree else
                       "Indeks tidak sepenuhnya sepakat; k dipilih memakai aturan Silhouette di atas"
                       + (f" (k = {K} juga unggul pada {', '.join(n for n, v in best_k.items() if v == K and n != 'Silhouette')})."
                          if any(v == K and n != "Silhouette" for n, v in best_k.items()) else ".")))
        t = kms.copy()
        t["dipilih"] = np.where(t["k"] == K, "✓", "")
        t["min_cluster_share"] = (t["min_cluster_share"] * 100).round(1)
        ui.dataframe(t.rename(columns={"min_cluster_share": "cluster terkecil (%)"}).round(4), hide_index=True)

        ui.section("Perbandingan dengan metode lain", "Semua metode dibandingkan pada k yang sama.")
        mc = mcmp.copy()
        mc.columns = ["Metode", "k", "Silhouette (ruang sendiri)", "Silhouette (Gower, sampel)", "Stabilitas ARI (rata-rata)",
                      "Stabilitas ARI (SD)", "Cluster terkecil (%)", "Selisih churn antar-cluster (pp, pasca-hoc)", "p chi-square churn (pasca-hoc)"]
        mc["p chi-square churn (pasca-hoc)"] = mc["p chi-square churn (pasca-hoc)"].map(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}")
        ui.dataframe(mc.round(4), hide_index=True)
        st.caption("Silhouette pada ruang fitur sendiri tidak bisa dibandingkan dengan K-Prototypes dan K-Medoids (ruang berbeda), sehingga "
                   "disediakan Silhouette pada ruang jarak Gower di sampel 3.000 pelanggan. Stabilitas ARI: dua subsampel 70% di-fit "
                   "terpisah lalu dibandingkan (1 = identik). Kolom churn bersifat pasca-hoc dan tidak dipakai memilih metode.")

        hb = ext.get("hierarchical_by_k"); lk = ext.get("linkage_comparison")
        with st.expander("Hierarchical Clustering (Ward, average, complete)"):
            if lk:
                st.markdown("**Perbandingan linkage** (sampel 2.500 pelanggan, pada k yang dipilih)")
                ui.dataframe(pd.DataFrame(lk).rename(columns={"linkage": "Linkage", "cophenetic_corr": "Korelasi cophenetic",
                                                              "silhouette_at_k": "Silhouette", "min_cluster_share": "Cluster terkecil"}),
                             hide_index=True)
            if hb:
                st.markdown("**Ward menurut k** (seluruh pelanggan)")
                ui.dataframe(pd.DataFrame(hb).rename(columns={"k": "k", "silhouette": "Silhouette", "davies_bouldin": "Davies-Bouldin",
                                                              "calinski_harabasz": "Calinski-Harabasz",
                                                              "min_cluster_share": "Cluster terkecil"}), hide_index=True)
            ui.image(R / "segmentation" / "06_dendrogram.png", "Dendrogram Ward (dari notebook)")
        gm = ext.get("gmm")
        with st.expander("Gaussian Mixture Model (AIC / BIC)"):
            if gm:
                g = pd.DataFrame(gm)
                fig = go.Figure()
                for cov, col in (("full", TEAL), ("diag", CORAL)):
                    s = g[g["cov"] == cov]
                    fig.add_trace(go.Scatter(x=s["k"], y=s["BIC"], mode="lines+markers", name=f"BIC ({cov})", line=dict(color=col)))
                fig.update_xaxes(title="Jumlah komponen", dtick=1); fig.update_yaxes(title="BIC")
                ui.plot(style_fig(fig, 300, True, "BIC GMM menurut jumlah komponen"), key="m_gmm")
                low = ext.get("gmm_lowest_bic", {})
                st.markdown(f"BIC terendah ada pada k = **{low.get('k')}** (kovarians `{low.get('cov')}`) dan terus menurun seiring k bertambah, "
                            "sehingga BIC tidak memberi titik optimal yang jelas pada data ini. Fitur `contract_ord` dan `n_addon_services` bersifat "
                            "diskret, sehingga asumsi Gaussian GMM hanya pendekatan.")
                ui.dataframe(g.rename(columns={"cov": "Kovarians", "mean_max_posterior": "Rata-rata posterior maks.",
                                               "min_cluster_share": "Cluster terkecil"}).round(3), hide_index=True)

        ui.section("Alasan pemilihan model final")
        km = mcmp[mcmp["metode"] == "K-Means"].iloc[0]; gmr = mcmp[mcmp["metode"] == "GMM"].iloc[0]
        sil_ok = gmr["silhouette_ruang_sendiri"] > km["silhouette_ruang_sendiri"]
        stab_ok = gmr["stabilitas_ARI_mean"] > km["stabilitas_ARI_mean"]
        with st.container(border=True):
            st.markdown(f"**Metode final: {str(summ.get('final_method', '')).upper().replace('KMEANS', 'K-Means')}, k = {K}**")
            st.markdown(
                "- **Aturan di notebook:** GMM dipilih hanya jika Silhouette **dan** stabilitasnya lebih tinggi daripada K-Means; selain itu K-Means "
                "dipakai karena lebih sederhana dan mudah diinterpretasi.\n"
                f"- **Silhouette:** K-Means {km['silhouette_ruang_sendiri']:.4f} vs GMM {gmr['silhouette_ruang_sendiri']:.4f} "
                f"({'GMM lebih tinggi' if sil_ok else 'K-Means lebih tinggi'}).\n"
                f"- **Stabilitas ARI:** K-Means {km['stabilitas_ARI_mean']:.4f} vs GMM {gmr['stabilitas_ARI_mean']:.4f} "
                f"({'GMM lebih tinggi' if stab_ok else 'K-Means lebih tinggi'}).\n"
                f"- **Hasil aturan:** {'kedua syarat GMM terpenuhi' if (sil_ok and stab_ok) else 'syarat GMM tidak terpenuhi, sehingga K-Means dipilih'}.\n"
                f"- **Catatan jujur:** pada Silhouette ruang Gower, GMM ({gmr['silhouette_gower_sampel']:.4f}) lebih tinggi daripada K-Means "
                f"({km['silhouette_gower_sampel']:.4f}); indikator ini tidak termasuk aturan pemilihan. Kedua metode sama-sama stabil (ARI di atas 0,98).\n"
                "- **Dapat dideploy:** hanya K-Means dan GMM yang dapat memprediksi segmen pelanggan baru. Hierarchical, K-Prototypes, dan K-Medoids "
                "dipakai sebagai pembanding robustness.")
        with st.expander("Gambar dari notebook"):
            c1, c2 = st.columns(2)
            with c1:
                ui.image(R / "segmentation" / "06_kmeans_k_selection.png", "Pemilihan k (K-Means)")
            with c2:
                ui.image(R / "segmentation" / "06_profil_segmen.png", "Profil segmen")

# ====================================================================================== C. PREDIKSI CHURN
with tabs[2]:
    if cvdf is None or not BEST:
        missing("cv_model_comparison.csv / metrics_summary.json")
    else:
        sp = ext.get("split", {}); tu = ext.get("tuning", {}); ts = ext.get("threshold_selection", {})
        ui.section("Rancangan eksperimen")
        st.markdown(
            f"- **Split:** stratified {int((1 - sp.get('test_size', 0.2)) * 100)}/{int(sp.get('test_size', 0.2) * 100)} "
            f"(random_state = {sp.get('random_state')}): **{sp.get('n_train', '?'):,}** data latih dan **{sp.get('n_test', '?'):,}** data uji, "
            f"churn {sp.get('churn_train_pct')}% pada keduanya. Test set tidak dipakai memilih model.\n"
            "- **Preprocessing di dalam Pipeline:** imputer median + StandardScaler untuk numerik, OneHotEncoder untuk kategorikal, "
            "sehingga hanya dipelajari dari data latih (mencegah kebocoran data).\n"
            f"- **Cross-validation:** {tu.get('cv_folds', 5)}-fold stratified pada data latih.\n"
            f"- **Tuning:** RandomizedSearchCV ({tu.get('n_iter_randomized_search', '?')} kombinasi per model) yang mengoptimalkan "
            f"{tu.get('scoring_for_selection', 'PR-AUC')}, metrik yang lebih sensitif terhadap ketidakseimbangan kelas daripada accuracy.\n"
            "- **Model yang dibandingkan:** Logistic Regression (tanpa bobot, class_weight balanced, dan SMOTE di dalam fold), Decision Tree, "
            f"Random Forest, Gradient Boosting, dan XGBoost (scale_pos_weight = {ext.get('scale_pos_weight', '?')}).\n"
            "- **Pemilihan model:** rata-rata PR-AUC cross-validation, bukan hasil test set.")

        ui.section("Perbandingan model (cross-validation)")
        cv = cvdf.sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
        tbl = pd.DataFrame({"Model": cv["model"], "PR-AUC": [pm(a, b) for a, b in zip(cv["cv_pr_auc"], cv["cv_pr_auc_sd"])],
                            "ROC-AUC": [pm(a, b) for a, b in zip(cv["cv_roc_auc"], cv["cv_roc_auc_sd"])],
                            "F1": [pm(a, b) for a, b in zip(cv["cv_f1"], cv["cv_f1_sd"])],
                            "Precision": [pm(a, b) for a, b in zip(cv["cv_precision"], cv["cv_precision_sd"])],
                            "Recall": [pm(a, b) for a, b in zip(cv["cv_recall"], cv["cv_recall_sd"])],
                            "Accuracy": [pm(a, b) for a, b in zip(cv["cv_accuracy"], cv["cv_accuracy_sd"])],
                            "Terpilih": np.where(cv["model"] == BEST, "✓", "")})
        ui.dataframe(tbl, hide_index=True)
        ui.plot(cv_bar(cv, "cv_pr_auc", "cv_pr_auc_sd", "model", BEST, "PR-AUC cross-validation per model (merah = terpilih)"), key="m_cvbar")
        top = cv.iloc[0]
        near = cv[cv["cv_pr_auc"] >= top["cv_pr_auc"] - top["cv_pr_auc_sd"]]["model"].tolist()
        with st.container(border=True):
            st.markdown(f"**Model terpilih: {BEST}** (PR-AUC CV {top['cv_pr_auc']:.4f}, ROC-AUC CV {top['cv_roc_auc']:.4f}).")
            st.markdown(f"Selisihnya dengan model lain kecil: **{len(near)} dari {len(cv)}** model berada dalam 1 SD dari yang terbaik "
                        f"({', '.join(near)}). Artinya pemilihan XGBoost mengikuti aturan skor tertinggi, tetapi model sederhana seperti "
                        "Logistic Regression hampir setara secara statistik.")
        st.caption("Skor CV dari pencarian hiperparameter sedikit optimistis karena tuning dan evaluasi memakai fold yang sama.")

        bp = cv[cv["model"] == BEST]["best_params"].iloc[0]
        params = {m.group(1): float(m.group(2)) for m in re.finditer(r"'clf__(\w+)': (?:np\.float64\()?(-?[\d.eE+-]+)\)?", str(bp))}
        if params:
            with st.expander(f"Hiperparameter terpilih ({BEST})"):
                ui.dataframe(pd.DataFrame({"Hiperparameter": list(params), "Nilai": [round(v, 4) for v in params.values()]}), hide_index=True)

        ui.section("Pemilihan threshold klasifikasi")
        if ts:
            st.markdown(
                f"Threshold dipilih dari prediksi **out-of-fold pada data latih** dengan kriteria **F{int(ts.get('beta', 2))}** "
                f"(recall dibobot {int(ts.get('beta', 2))}× precision), mencerminkan asumsi bahwa kehilangan pelanggan lebih mahal daripada "
                f"penawaran retensi yang tidak perlu. Hasilnya **{ts['threshold_f2']:.2f}** (precision OOF {ts['oof_precision']:.3f}, "
                f"recall OOF {ts['oof_recall']:.3f}). Sebagai pembanding, threshold yang memaksimalkan F1 adalah {ts['threshold_f1']:.2f}.")
        ui.image(R / "figures" / "08_threshold.png", "Trade-off precision/recall menurut threshold (dari notebook)")

# ====================================================================================== D. EVALUASI
with tabs[3]:
    if not summ.get("test_metrics"):
        missing("metrics_summary.json")
    else:
        tm = summ["test_metrics"]; k05 = tm["@0.5"]; kf2 = [v for k, v in tm.items() if k != "@0.5"][0]
        base = ext.get("baseline_pr_auc_test"); ci = summ.get("bootstrap_ci_95", {})
        sp = ext.get("split", {})
        ui.section(f"Hasil model final ({BEST}) pada test set",
                   f"{sp.get('n_test', '?'):,} pelanggan yang tidak dipakai untuk memilih model. Dilaporkan satu kali.")
        c1, c2 = st.columns(2)
        for col, title, m in ((c1, "Threshold 0.50", k05), (c2, f"Threshold {THR:.2f} (F2, dipakai aplikasi)", kf2)):
            with col:
                st.markdown(f"**{title}**")
                ui.kpi_row([("Accuracy", f"{m['accuracy']:.3f}", "", "#64748B"), ("Precision", f"{m['precision']:.3f}", "", "#0F6E8C"),
                            ("Recall", f"{m['recall']:.3f}", "", "#D1495B"), ("F1", f"{m['f1']:.3f}", "", "#F2A541")])
        st.write("")
        ui.kpi_row([("ROC-AUC", f"{k05['roc_auc']:.3f}", f"CI 95%: {ci['roc_auc'][0]:.3f} - {ci['roc_auc'][1]:.3f}" if ci else "", "#14213D"),
                    ("PR-AUC", f"{k05['pr_auc']:.3f}", f"CI 95%: {ci['pr_auc'][0]:.3f} - {ci['pr_auc'][1]:.3f}" if ci else "", "#14213D"),
                    ("Baseline PR-AUC", f"{base:.3f}" if base else "n/a", "= prevalensi churn di test", "#64748B")])
        st.caption("ROC-AUC dan PR-AUC tidak bergantung pada threshold. CI 95% dari bootstrap 500 kali pada test set.")

        tp = load_test_predictions()
        interactive = bool(tp and tp["ok"])
        if tp and not tp["ok"]:
            st.warning("File prediksi test ditemukan tetapi tidak lolos validasi, sehingga versi statis ditampilkan: "
                       + " ".join(tp["problems"]))

        if interactive:
            df, models = tp["df"], tp["models"]
            y = df["y_true"].values
            ui.section("Kurva evaluasi interaktif", "Dihitung dari prediksi test set yang diekspor notebook (probabilitas tiap model).")
            sel = st.multiselect("Model yang ditampilkan", models, default=models, key="m_curve_models")
            if sel:
                roc_c = {m: (*roc_points(y, df[m].values), roc_pr_summary(y, df[m].values)["roc_auc"]) for m in sel}
                pr_c = {m: (*pr_points(y, df[m].values), roc_pr_summary(y, df[m].values)["pr_auc"]) for m in sel}
                a, b = st.columns(2)
                with a:
                    ui.plot(roc_figure(roc_c, BEST if BEST in sel else None), key="m_roc")
                with b:
                    ui.plot(pr_figure(pr_c, float(y.mean()), BEST if BEST in sel else None), key="m_pr")
            ui.section("Confusion matrix dan threshold", "Geser threshold untuk melihat trade-off precision dan recall pada test set.")
            mdl = st.selectbox("Model", models, index=models.index(BEST) if BEST in models else 0, key="m_cm_model")
            thr = st.slider("Threshold", 0.05, 0.95, float(round(THR, 2)), 0.01, key="m_thr")
            m = metrics_at(y, df[mdl].values, thr)
            a, b = st.columns([1, 1.2])
            with a:
                ui.plot(confusion_figure(m["tn"], m["fp"], m["fn"], m["tp"], f"Confusion matrix @ {thr:.2f}"), key="m_cm")
            with b:
                ui.plot(threshold_figure(threshold_table(y, df[mdl].values), thr), key="m_thrfig")
            ui.kpi_row([("Accuracy", f"{m['accuracy']:.3f}", "", "#64748B"), ("Precision", f"{m['precision']:.3f}", "", "#0F6E8C"),
                        ("Recall", f"{m['recall']:.3f}", "", "#D1495B"), ("F1", f"{m['f1']:.3f}", "", "#F2A541"),
                        ("F2", f"{m['f_beta']:.3f}", "", "#6C5B9E")])
            st.markdown(f"Pada threshold {thr:.2f}, model menangkap **{m['recall'] * 100:.1f}%** churner ({m['tp']:,} dari {m['tp'] + m['fn']:,}), "
                        f"dengan **{m['fp']:,}** pelanggan non-churn yang ikut ditawari retensi.")
            if mdl == BEST and abs(thr - round(THR, 2)) < 1e-9:
                ok = abs(m["recall"] - kf2["recall"]) < 5e-3 and abs(m["precision"] - kf2["precision"]) < 5e-3
                st.caption("Konsistensi dengan notebook pada threshold F2: " + ("cocok." if ok else "TIDAK cocok, periksa file prediksi."))
        else:
            cm = ext.get("confusion_matrix_at_f2_threshold")
            ui.section("Confusion matrix dan kurva (versi statis)")
            if cm:
                a, b = st.columns([1, 1.3])
                with a:
                    ui.plot(confusion_figure(cm["TN"], cm["FP"], cm["FN"], cm["TP"], f"Confusion matrix @ {THR:.2f} (dari notebook)"),
                            key="m_cm_static")
                with b:
                    rec = cm["TP"] / (cm["TP"] + cm["FN"])
                    st.markdown(f"- **TP = {cm['TP']}** churner tertangkap, **FN = {cm['FN']}** terlewat.\n"
                                f"- **FP = {cm['FP']}** non-churner ikut ditawari retensi, **TN = {cm['TN']}**.\n"
                                f"- Recall **{rec * 100:.1f}%**.")
            ui.image(R / "figures" / "08_evaluasi_test.png", "ROC, Precision-Recall, dan confusion matrix (dari notebook)")
            ui.note("Grafik interaktif (kurva ROC/PR yang bisa disorot dan threshold yang bisa digeser) akan aktif otomatis setelah file "
                    "<code>test_predictions.csv</code> dan <code>test_predictions_meta.json</code> diletakkan di folder "
                    "<code>reports/metrics/</code>. Kode sel Colab untuk mengekspornya ada di <code>tools/colab_export_test_predictions.py</code>.")

        if testdf is not None:
            ui.section("Semua model pada test set (threshold 0.50)")
            ui.dataframe(testdf.sort_values("pr_auc", ascending=False).rename(columns={
                "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}).round(4))
            st.caption("Pada threshold 0,50, accuracy tinggi pada model tanpa pembobotan kelas terjadi karena recall-nya rendah; "
                       "karena itu accuracy tidak dijadikan kriteria utama.")
        with st.expander("Kalibrasi probabilitas"):
            ui.image(R / "figures" / "08_kalibrasi.png", "Kurva kalibrasi pada test set (dari notebook)")
            st.markdown("Model berbobot kelas cenderung melebih-lebihkan probabilitas. Karena itu aplikasi menampilkan **probabilitas "
                        "terkalibrasi** (regresi isotonik pada skor out-of-fold), sedangkan tingkat risiko ditentukan dari skor mentah.")

# ====================================================================================== E. INTERPRETABILITAS
with tabs[4]:
    nat = load_report_csv("interpretability/native_importance.csv")
    per = load_report_csv("interpretability/permutation_importance.csv")
    sh = load_report_csv("interpretability/shap_importance.csv")
    if nat is None or per is None:
        missing("native_importance.csv / permutation_importance.csv")
    else:
        st.markdown("Tiga sudut pandang dipakai agar kesimpulan tidak bergantung pada satu metode. Fitur one-hot digabung kembali ke "
                    "fitur asalnya. Semuanya menggambarkan **kontribusi pada prediksi model**, bukan sebab-akibat.")
        nat = nat.rename(columns={nat.columns[0]: "fitur"})
        series = {"Native (importance bawaan)": nat.set_index("fitur")["native_importance"],
                  "Permutation (penurunan PR-AUC)": per.set_index("fitur")["perm_mean"]}
        if sh is not None:
            sh = sh.rename(columns={sh.columns[0]: "fitur"})
            series["SHAP (rata-rata |nilai|)"] = sh.set_index("fitur")["mean_abs_shap"]
        cols = st.columns(len(series))
        for col, (name, s) in zip(cols, series.items()):
            with col:
                top = s.sort_values(ascending=False).head(10)
                ui.plot(hbar(top.index.tolist(), top.values.tolist(), TEAL, height=380, fmt="{:.3f}", title=name), key=f"m_imp_{name[:6]}")
        SHORT = {"Native (importance bawaan)": "Native", "Permutation (penurunan PR-AUC)": "Permutation", "SHAP (rata-rata |nilai|)": "SHAP"}
        ranks = pd.DataFrame({SHORT[k]: v.rank(ascending=False) for k, v in series.items()})
        cons = ranks.assign(rata2=ranks.mean(axis=1)).sort_values("rata2").head(10)
        a, b = st.columns(2)
        with a:
            st.markdown("**10 fitur teratas menurut rata-rata peringkat**")
            ui.dataframe(cons.rename(columns={"rata2": "Rata-rata"}).round(1))
        with b:
            st.markdown("**Korelasi peringkat (Spearman) antar metode**")
            ui.dataframe(ranks.corr(method="spearman").round(3))
        first = {SHORT[k]: v.sort_values(ascending=False).index[0] for k, v in series.items()}
        st.markdown("**Fitur teratas:** " + "; ".join(f"{k} → **{v}**" for k, v in first.items()) + ".")
        ui.note("Metode berbeda dapat menempatkan fitur yang saling berkorelasi (mis. <b>tenure</b>, <b>TotalCharges</b>, "
                "<b>MonthlyCharges</b>, dan <b>Contract</b>) pada peringkat yang berbeda, karena pentingnya terbagi di antara fitur "
                "yang membawa informasi serupa. Baca ketiganya bersama, bukan satu per satu.", warn=True)
        if sh is not None:
            with st.expander("SHAP beeswarm dan importance dari notebook"):
                ui.image(R / "interpretability" / "09_shap_beeswarm.png", "SHAP beeswarm (800 sampel test)")
                ui.image(R / "interpretability" / "09_importance.png", "Native dan permutation importance")
        else:
            st.info("Hasil SHAP tidak tersedia di artefak, sehingga hanya importance bawaan dan permutation yang ditampilkan.")

# ====================================================================================== F. KETERBATASAN
with tabs[5]:
    rc = summ.get("risk_cutoffs", {})
    ci = summ.get("bootstrap_ci_95", {})
    ui.section("Keterbatasan dataset")
    st.markdown(
        "- **Satu snapshot tanpa tanggal:** tidak ada urutan waktu, musiman, atau definisi kapan churn terjadi. Analisis survival mengandalkan "
        "asumsi bahwa pelanggan yang belum churn tersensor pada tenure saat ini.\n"
        "- **Tidak ada data biaya, margin, keluhan, kualitas jaringan, atau penawaran pesaing.** `annual_value_proxy` (tagihan bulanan × 12) hanya "
        "proksi pendapatan, bukan nilai umur pelanggan (CLV).\n"
        "- **Satu operator dan wilayah yang tidak dijelaskan;** generalisasi ke populasi lain terbatas.\n"
        "- **MonthlyCharges adalah tagihan saat ini,** sedangkan TotalCharges akumulasi historis, sehingga keduanya redundan dengan tenure.")
    ui.section("Keterbatasan model")
    lines = [
        "- **Ketidakpastian estimasi:** " + (f"CI 95% bootstrap ROC-AUC {ci['roc_auc'][0]:.3f} sampai {ci['roc_auc'][1]:.3f} dan PR-AUC "
                                            f"{ci['pr_auc'][0]:.3f} sampai {ci['pr_auc'][1]:.3f} pada test set." if ci else "tidak tersedia."),
        "- **Model sebanding:** beberapa model memiliki skor CV yang sulit dibedakan secara statistik (lihat bagian C).",
        "- **Threshold bergantung asumsi biaya:** F2 mengasumsikan kehilangan pelanggan lebih mahal daripada penawaran yang tidak perlu. "
        "Biaya sebenarnya tidak diketahui; threshold optimal sebaiknya dipilih ulang dengan model biaya eksplisit.",
        "- **Skor pelanggan yang ada di data:** model final dilatih pada seluruh data, sehingga skor in-sample lebih optimistis daripada skor "
        "out-of-fold. Dashboard memakai skor out-of-fold untuk dataset bawaan; simulator memakai model final.",
        "- **Probabilitas perlu dipantau:** kalibrasi isotonik dilakukan pada skor out-of-fold dan dapat bergeser pada populasi baru.",
        "- **Pelanggan di luar rentang data latih** (tenure di atas 72 bulan atau tagihan di luar 18,25 sampai 118,75) diprediksi kurang andal.",
    ]
    st.markdown("\n".join(lines))
    ui.section("Asumsi analitis")
    st.markdown(
        (f"- **Tingkat risiko** memakai kuantil skor: High jika di atas kuantil {rc.get('quantiles', [0.5, 0.8])[1]:.2f} "
         f"(skor mentah ≥ {rc.get('high', 0):.3f}), Medium di atas kuantil {rc.get('quantiles', [0.5, 0.8])[0]:.2f} "
         f"(≥ {rc.get('medium', 0):.3f}). Ini keputusan analis, bukan hasil data, sehingga proporsi High pada dataset bawaan selalu sekitar 20%.\n"
         if rc else "")
        + (f"- **Nilai tinggi** = kuartil teratas proksi nilai tahunan (≥ ${summ.get('value_cutoff', 0):,.0f}).\n" if summ.get("value_cutoff") else "")
        + "- **Prioritas P1 sampai P4** adalah kombinasi tingkat risiko dan nilai; panduan tindakan per prioritas adalah aturan, bukan keluaran model.\n"
        "- **Nama segmen** dibuat otomatis dari tercile tenure dan tagihan, sehingga label seperti \"Menengah\" dapat mencakup rentang luas.\n"
        "- **Rekomendasi retensi** dibangkitkan dari aturan berbasis profil segmen dan harus diperlakukan sebagai hipotesis.")
    ui.section("Asosiasi bukan sebab-akibat")
    ui.note("Seluruh temuan, termasuk uji statistik, odds ratio, importance, dan simulasi skenario, menggambarkan <b>pola asosiasi</b> pada data "
            "observasional. Misalnya, pelanggan berkontrak dua tahun jarang churn, tetapi kontrak berjangka juga membatasi kapan pelanggan boleh "
            "berhenti; data ini tidak dapat menunjukkan bahwa mengubah kontrak akan <i>menyebabkan</i> penurunan churn. Efek intervensi retensi hanya "
            "dapat diukur dengan uji A/B atau uplift modeling.", warn=True)
    rep = R / "business" / "final_findings.md"
    if rep.exists():
        with st.expander("Laporan temuan otomatis dari notebook (final_findings.md)"):
            st.markdown(rep.read_text(encoding="utf-8"))
