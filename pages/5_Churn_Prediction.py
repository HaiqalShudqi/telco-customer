"""Halaman 5: Churn Prediction Simulator (memakai model tersimpan; tidak ada training ulang)."""
import pandas as pd
import streamlit as st

from utils import simulator as sim
from utils import ui
from utils.analytics import segment_profile
from utils.business import PRIORITY_ACTIONS, PRIORITY_DEFINITION
from utils.data_loader import load_report_csv, load_scored_default
from utils.model_loader import ModelLoadError, load_model_bundle, load_schema
from utils.visualizations import RISK_COLORS, prob_gauge, risk_zone_bar, seg_short

ui.bootstrap()
ui.get_dataset()  # memastikan sidebar dan sumber data terpasang; simulator sendiri tidak bergantung pada dataset aktif
ui.page_header("Churn Prediction Simulator",
               "Masukkan profil pelanggan baru untuk memperoleh probabilitas churn, segmen, tingkat risiko, dan prioritas retensi.")

try:
    bundle = load_model_bundle()
    schema = load_schema()
except ModelLoadError as exc:
    st.error(f"Simulator tidak dapat dipakai: {exc}")
    st.stop()

CAT = schema["categorical_allowed_values"]
YN = {"Yes": "Ya", "No": "Tidak"}
ADDON_LABEL = {"OnlineSecurity": "Online Security", "OnlineBackup": "Online Backup", "DeviceProtection": "Device Protection",
               "TechSupport": "Tech Support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming Movies"}

DEFAULTS = {
    "sim_gender": "Female", "sim_senior": 0, "sim_partner": "No", "sim_dep": "No", "sim_tenure": 12,
    "sim_phone": "Yes", "sim_ml": "No", "sim_inet": "DSL", "sim_OnlineSecurity": "No", "sim_OnlineBackup": "No",
    "sim_DeviceProtection": "No", "sim_TechSupport": "No", "sim_StreamingTV": "No", "sim_StreamingMovies": "No",
    "sim_contract": "Month-to-month", "sim_paperless": "Yes", "sim_pay": "Electronic check", "sim_monthly": 70.0,
    "sim_auto_total": True, "sim_total": 840.0,
}
PRESETS = {
    "Pelanggan baru, fiber optic, kontrak bulanan": dict(
        sim_tenure=2, sim_phone="Yes", sim_ml="No", sim_inet="Fiber optic", sim_contract="Month-to-month",
        sim_pay="Electronic check", sim_paperless="Yes", sim_monthly=85.0, sim_partner="No", sim_dep="No",
        sim_OnlineSecurity="No", sim_OnlineBackup="No", sim_DeviceProtection="No", sim_TechSupport="No",
        sim_StreamingTV="No", sim_StreamingMovies="No"),
    "Pelanggan lama, kontrak 2 tahun, banyak layanan": dict(
        sim_tenure=60, sim_phone="Yes", sim_ml="Yes", sim_inet="DSL", sim_contract="Two year",
        sim_pay="Credit card (automatic)", sim_paperless="No", sim_monthly=75.0, sim_partner="Yes", sim_dep="Yes",
        sim_OnlineSecurity="Yes", sim_OnlineBackup="Yes", sim_DeviceProtection="Yes", sim_TechSupport="Yes",
        sim_StreamingTV="No", sim_StreamingMovies="No"),
    "Hanya telepon, tanpa internet": dict(
        sim_tenure=24, sim_phone="Yes", sim_ml="No", sim_inet="No", sim_contract="One year",
        sim_pay="Mailed check", sim_paperless="No", sim_monthly=25.0, sim_partner="Yes", sim_dep="No"),
}


def _apply(values: dict) -> None:
    st.session_state.update(values)
    st.session_state["sim_auto_total"] = True


def _reset() -> None:
    _apply(DEFAULTS)
    st.session_state.pop("sim_result", None)


for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ------------------------------------------------------------------ Contoh cepat
ui.section("Isi formulir", "Pilih contoh untuk mengisi formulir dengan cepat, atau isi manual. Kolom bertanda hanya dapat dipilih sesuai logika layanan.")
pc = st.columns(len(PRESETS) + 1)
for i, (col, (label, vals)) in enumerate(zip(pc, PRESETS.items())):
    with col:
        ui.stretch_button(label, on_click=_apply, args=({**DEFAULTS, **vals},), key=f"preset_{i}")
with pc[-1]:
    ui.stretch_button("Kosongkan formulir", on_click=_reset, key="sim_reset")

# ------------------------------------------------------------------ Formulir
f1, f2, f3 = st.columns(3)
with f1:
    st.markdown("**Profil pelanggan**")
    st.selectbox("Gender", CAT["gender"], key="sim_gender")
    st.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Ya" if v == 1 else "Tidak", key="sim_senior")
    st.selectbox("Partner", ["Yes", "No"], format_func=YN.get, key="sim_partner")
    st.selectbox("Tanggungan (Dependents)", ["Yes", "No"], format_func=YN.get, key="sim_dep")
    st.number_input("Tenure (bulan)", min_value=0, max_value=120, step=1, key="sim_tenure",
                    help="Lama berlangganan. Data training mencakup 0 sampai 72 bulan.")

with f2:
    st.markdown("**Layanan**")
    st.selectbox("Layanan telepon", ["Yes", "No"], format_func=YN.get, key="sim_phone")
    if st.session_state["sim_phone"] == "No":
        st.session_state["sim_ml"] = "No phone service"
        ml_opts, ml_dis = ["No phone service"], True
    else:
        if st.session_state["sim_ml"] == "No phone service":
            st.session_state["sim_ml"] = "No"
        ml_opts, ml_dis = ["No", "Yes"], False
    st.selectbox("Multiple lines", ml_opts, key="sim_ml", disabled=ml_dis,
                 format_func=lambda v: {"Yes": "Ya", "No": "Tidak", "No phone service": "Tanpa layanan telepon"}[v])
    st.selectbox("Layanan internet", CAT["InternetService"], key="sim_inet",
                 format_func=lambda v: {"No": "Tidak ada"}.get(v, v))
    no_inet = st.session_state["sim_inet"] == "No"
    for a in sim.ADDONS:
        key = f"sim_{a}"
        if no_inet:
            st.session_state[key] = "No internet service"
            opts = ["No internet service"]
        else:
            if st.session_state[key] == "No internet service":
                st.session_state[key] = "No"
            opts = ["No", "Yes"]
        st.selectbox(ADDON_LABEL[a], opts, key=key, disabled=no_inet,
                     format_func=lambda v: {"Yes": "Ya", "No": "Tidak", "No internet service": "Tanpa internet"}[v])

with f3:
    st.markdown("**Kontrak dan tagihan**")
    st.selectbox("Jenis kontrak", CAT["Contract"], key="sim_contract")
    st.selectbox("Paperless billing", ["Yes", "No"], format_func=YN.get, key="sim_paperless")
    st.selectbox("Metode pembayaran", CAT["PaymentMethod"], key="sim_pay")
    st.number_input("Tagihan bulanan / MonthlyCharges ($)", min_value=0.0, max_value=300.0, step=0.05, format="%.2f", key="sim_monthly")
    auto = st.checkbox("Hitung TotalCharges otomatis (tenure × tagihan bulanan)", key="sim_auto_total")
    if auto:
        st.session_state["sim_total"] = sim.auto_total(st.session_state["sim_tenure"], st.session_state["sim_monthly"])
    st.number_input("Total tagihan / TotalCharges ($)", min_value=0.0, max_value=50000.0, step=10.0, format="%.2f",
                    key="sim_total", disabled=auto,
                    help="Aktifkan perhitungan otomatis, atau nonaktifkan untuk mengisi sendiri jika pelanggan pernah mengganti paket.")
    if auto:
        st.caption("Nilai awal hanya perkiraan. Total tagihan nyata dapat berbeda jika paket pernah berubah.")


def collect() -> dict:
    s = st.session_state
    return {"gender": s["sim_gender"], "SeniorCitizen": int(s["sim_senior"]), "Partner": s["sim_partner"],
            "Dependents": s["sim_dep"], "tenure": int(s["sim_tenure"]), "PhoneService": s["sim_phone"],
            "MultipleLines": s["sim_ml"], "InternetService": s["sim_inet"],
            **{a: s[f"sim_{a}"] for a in sim.ADDONS},
            "Contract": s["sim_contract"], "PaperlessBilling": s["sim_paperless"], "PaymentMethod": s["sim_pay"],
            "MonthlyCharges": float(s["sim_monthly"]), "TotalCharges": float(s["sim_total"])}


inputs = collect()
go = st.button("Prediksi churn", type="primary", key="sim_go")
if go:
    try:
        row, warns, valid = sim.predict_one(inputs)
        st.session_state["sim_result"] = {"inputs": dict(inputs), "row": row, "warnings": warns, "valid": valid}
    except sim.InputError as e:
        st.session_state.pop("sim_result", None)
        st.error("Input belum valid:\n\n" + "\n".join(f"- {m}" for m in e.errors))
    except ModelLoadError as e:
        st.session_state.pop("sim_result", None)
        st.error(str(e))

res = st.session_state.get("sim_result")
if not res:
    st.info("Isi formulir lalu tekan **Prediksi churn**.")
    st.stop()

if res["inputs"] != inputs:
    ui.note("Input telah diubah sejak prediksi terakhir. Tekan <b>Prediksi churn</b> lagi untuk memperbarui hasil.", warn=True)
for w in res["warnings"]:
    st.warning(w)

r = res["row"]
tier = r["risk_tier"]
ui.section("Hasil prediksi")
ui.kpi_row([
    ("Probabilitas churn", f"{r['churn_prob_calibrated'] * 100:.1f}%", "terkalibrasi", RISK_COLORS[tier]),
    ("Klasifikasi risiko", tier, f"skor mentah {r['churn_prob_raw']:.3f}", RISK_COLORS[tier]),
    ("Segmen", seg_short(r["segment_name"]), r["segment_name"].split(":", 1)[1].strip(), "#0F6E8C"),
    ("Tingkat nilai", r["value_tier"], f"proksi tahunan ${r['annual_value_proxy']:,.0f}", "#14213D"),
    ("Prioritas retensi", r["priority"].split(" - ")[0], r["priority"].split(" - ")[1], "#F2A541"),
])

g1, g2 = st.columns([1, 1.5])
with g1:
    ui.plot(prob_gauge(r["churn_prob_calibrated"] * 100, RISK_COLORS[tier]), key="sim_gauge")
with g2:
    ui.plot(risk_zone_bar(float(r["churn_prob_raw"]), bundle["risk_cut_med"], bundle["risk_cut_high"]), key="sim_zone")
    flag = int(r["flag_churn_threshold"]) == 1
    ref = load_scored_default()["churn_prob_raw"]
    pct = sim.score_percentile(float(r["churn_prob_raw"]), ref)
    st.markdown(
        f"- Pada threshold klasifikasi F2 (**{bundle['threshold']:.2f}**), pelanggan ini diprediksi **{'churn' if flag else 'tidak churn'}**.\n"
        f"- Skor mentah lebih tinggi dari sekitar **{pct:.0f}%** pelanggan pada dataset bawaan.\n"
        f"- Proksi *revenue at risk* pelanggan ini: **{r['revenue_at_risk']:,.1f}** (indeks relatif, bukan prakiraan kerugian).")

ui.section("Saran tindakan bisnis")
a1, a2 = st.columns(2)
with a1:
    with st.container(border=True):
        st.markdown(f"**{r['priority']}**")
        st.caption(PRIORITY_DEFINITION[r["priority"]])
        st.markdown(PRIORITY_ACTIONS[r["priority"]])
        st.caption("Panduan ini berbasis aturan prioritas, bukan keluaran model.")
with a2:
    with st.container(border=True):
        st.markdown(f"**Konteks segmen {seg_short(r['segment_name'])}**")
        scored = load_scored_default()
        prof = segment_profile(scored)
        sid = int(r["segment_id"])
        p = prof.loc[sid]
        st.markdown(f"Pada dataset bawaan, segmen ini berisi {int(p['n']):,} pelanggan ({p['share_%']:.1f}%), churn aktual "
                    f"**{p['churn_rate_%']:.1f}%**, tenure rata-rata {p['tenure_mean']:.0f} bulan.")
        rec = load_report_csv("business/business_recommendations.csv")
        if rec is not None and (rec["segment_id"] == sid).any():
            h = rec[rec["segment_id"] == sid].iloc[0]
            st.markdown("Rekomendasi segmen dari analisis notebook *(hipotesis, perlu uji A/B)*:")
            for act in str(h["rekomendasi"]).split(" | "):
                st.markdown(f"- {act}")

with st.expander("Simulasi skenario: bagaimana skor berubah jika satu faktor diganti?"):
    sc = sim.scenario_table(res["inputs"])
    if sc.empty:
        st.info("Skenario tidak dapat dihitung untuk input ini.")
    else:
        ui.dataframe(sc.round({"Probabilitas churn (%)": 1, "Skor mentah": 3, "Selisih vs saat ini (poin)": 1}), hide_index=True)
        ui.note("Ini <b>simulasi model</b> yang bergantung pada asosiasi dalam data, bukan estimasi efek intervensi yang sebenarnya. "
                "Tagihan bulanan dianggap tetap. Uji dengan A/B test sebelum dijadikan dasar kebijakan.", warn=True)

with st.expander("Detail teknis: input dan fitur yang dipakai model"):
    cf = bundle["_cf"]
    feats = cf.build_features(res["valid"])
    st.markdown("**Input mentah (setelah validasi)**")
    ui.dataframe(res["valid"].T.rename(columns={0: "nilai"}).astype(str))
    st.markdown("**Fitur turunan dari feature engineering notebook**")
    show = feats[["tenure_group", "n_addon_services", "total_service_count", "avg_monthly_revenue", "annual_value_proxy",
                  "contract_risk", "contract_ord"]].T.rename(columns={0: "nilai"})
    ui.dataframe(show.astype(str))
    st.caption(f"Batas risiko (skor mentah): Medium ≥ {bundle['risk_cut_med']:.3f}, High ≥ {bundle['risk_cut_high']:.3f}. "
               f"Nilai tinggi: proksi tahunan ≥ ${bundle['value_cut']:,.0f}. Model: {bundle['model_name']}.")

ui.note("Hasil ini berasal dari model yang dilatih di notebook dan dimuat apa adanya. Model memperkirakan probabilitas berdasarkan "
        "pola historis (asosiasi), bukan sebab-akibat. Jika Anda memasukkan pelanggan yang ada di dataset training, skor bisa sedikit "
        "berbeda dari skor out-of-fold di Customer Explorer karena model akhir dilatih pada seluruh data.")
