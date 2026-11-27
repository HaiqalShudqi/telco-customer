"""Komponen UI bersama: tema, header, kartu KPI, sidebar pemilihan data, dan akses dataset aktif."""
from __future__ import annotations

import inspect
import os
import sys
from html import escape

import pandas as pd
import streamlit as st

from utils import config
from utils.data_loader import DataLoadError, get_active_dataset, process_upload
from utils.model_loader import ModelLoadError, load_model_bundle

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', 'Segoe UI', Roboto, sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3 { color: #14213D; letter-spacing: -0.01em; }

/* Header halaman */
.page-hero { background: linear-gradient(120deg, #14213D 0%, #0F6E8C 100%); border-radius: 16px;
  padding: 1.35rem 1.6rem; margin-bottom: 1.2rem; color: #fff; }
.page-hero .eyebrow { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; opacity: .75; font-weight: 600; }
.page-hero h1 { color: #fff; font-size: 1.75rem; margin: .15rem 0 .25rem 0; padding: 0; }
.page-hero p { margin: 0; opacity: .88; font-size: .95rem; }

/* Kartu KPI */
.kpi { background: #fff; border: 1px solid #E5E9F0; border-radius: 14px; padding: .95rem 1.05rem;
  box-shadow: 0 1px 2px rgba(20,33,61,.04); min-height: 8.4rem; border-top: 3px solid var(--accent, #0F6E8C); }
.kpi .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #64748B; font-weight: 600; }
.kpi .value { font-size: 1.75rem; font-weight: 700; color: #14213D; line-height: 1.25; margin-top: .15rem; }
.kpi .sub { font-size: .78rem; color: #64748B; margin-top: .1rem; min-height: 1.1rem; }

/* Panel dan catatan */
.section-title { font-size: 1.05rem; font-weight: 600; color: #14213D; margin: 1.3rem 0 .15rem 0; }
.section-cap { font-size: .85rem; color: #64748B; margin-bottom: .5rem; }
.note { background: #EEF5F8; border-left: 4px solid #0F6E8C; border-radius: 8px; padding: .7rem .95rem;
  font-size: .86rem; color: #14213D; }
.note.warn { background: #FFF6E5; border-left-color: #F2A541; }
.insight { background: #fff; border: 1px solid #E5E9F0; border-radius: 14px; padding: .4rem 1.2rem; }
.insight li { margin: .55rem 0; line-height: 1.55; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #E5E9F0; }
.side-brand { font-weight: 700; color: #14213D; font-size: .95rem; letter-spacing: .02em; }
.side-sub { font-size: .72rem; color: #64748B; margin-bottom: .4rem; }
.side-card { background: #F6F8FB; border: 1px solid #E5E9F0; border-radius: 10px; padding: .6rem .75rem;
  font-size: .78rem; color: #14213D; line-height: 1.5; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str = "Telco Customer Intelligence") -> None:
    st.markdown(
        f'<div class="page-hero"><div class="eyebrow">{escape(eyebrow)}</div>'
        f"<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>", unsafe_allow_html=True)


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="section-cap">{escape(caption)}</div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    """Catatan berformat. `text` boleh berisi HTML sederhana yang dibuat oleh kode (bukan input pengguna)."""
    st.markdown(f'<div class="note{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", accent: str = "#0F6E8C") -> None:
    st.markdown(
        f'<div class="kpi" style="--accent:{accent}"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div><div class="sub">{escape(sub)}</div></div>',
        unsafe_allow_html=True)


def kpi_row(items: list[tuple]) -> None:
    """items: (label, value, sub, accent)"""
    cols = st.columns(len(items))
    for col, it in zip(cols, items):
        with col:
            kpi_card(*it)


_PLOT_WIDTH_ARG = "width" in inspect.signature(st.plotly_chart).parameters


def plot(fig, key: str | None = None) -> None:
    """st.plotly_chart yang kompatibel lintas versi Streamlit."""
    cfg = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
    if _PLOT_WIDTH_ARG:
        st.plotly_chart(fig, width="stretch", key=key, config=cfg)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config=cfg)


_DF_WIDTH_ARG = "width" in inspect.signature(st.dataframe).parameters


_BTN_WIDTH_ARG = "width" in inspect.signature(st.button).parameters


def stretch_button(label: str, **kwargs):
    """st.button selebar kolom, kompatibel lintas versi Streamlit."""
    if _BTN_WIDTH_ARG:
        return st.button(label, width="stretch", **kwargs)
    return st.button(label, use_container_width=True, **kwargs)


_IMG_WIDTH_ARG = "width" in inspect.signature(st.image).parameters


def image(path, caption: str | None = None) -> None:
    """Tampilkan gambar dari folder reports bila ada; kompatibel lintas versi Streamlit."""
    if not os.path.exists(str(path)):
        st.caption(f"(Gambar tidak ditemukan: {os.path.basename(str(path))})")
        return
    if _IMG_WIDTH_ARG:
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.image(str(path), caption=caption, use_container_width=True)


def dataframe(data, **kwargs):
    """st.dataframe yang kompatibel lintas versi Streamlit (lebar penuh)."""
    if _DF_WIDTH_ARG:
        return st.dataframe(data, width="stretch", **kwargs)
    return st.dataframe(data, use_container_width=True, **kwargs)


# ----------------------------------------------------------------------------------------------
# Dataset aktif (dipilih di sidebar, dipakai semua halaman)
# ----------------------------------------------------------------------------------------------
DEFAULT_LABEL = "Dataset bawaan (IBM Telco)"
UPLOAD_LABEL = "Unggah CSV sendiri"


def _resolve_dataset() -> dict:
    """Tentukan dataset aktif dari pilihan sidebar. Selalu mengembalikan dict yang aman dipakai halaman."""
    mode = st.session_state.get("data_mode", DEFAULT_LABEL)
    info = {"mode": "default", "table": None, "label": "", "warnings": [], "errors": [], "error_fatal": None}

    if mode == UPLOAD_LABEL:
        up = st.session_state.get("upload_file")
        if up is not None:
            res = process_upload(up.getvalue())
            if res["ok"]:
                info.update(mode="upload", table=res["table"], warnings=res["warnings"],
                            label=f"Dataset unggahan: {up.name} ({len(res['table']):,} pelanggan)")
                return info
            info["errors"] = res["errors"]
            info["warnings"] = res["warnings"]

    try:
        table, label, _ = get_active_dataset("default")
        info.update(table=table, label=label)
    except DataLoadError as exc:
        info["error_fatal"] = str(exc)
    return info


def render_sidebar() -> dict:
    """Gambar sidebar (pemilihan sumber data + status model) dan simpan dataset aktif di session_state."""
    with st.sidebar:
        st.markdown('<div class="side-brand">TELCO CUSTOMER INTELLIGENCE</div>'
                    '<div class="side-sub">Analytics · Segmentation · Churn</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("**Sumber data**")
        st.radio("Pilih sumber data", [DEFAULT_LABEL, UPLOAD_LABEL], key="data_mode", label_visibility="collapsed")
        if st.session_state.get("data_mode") == UPLOAD_LABEL:
            st.file_uploader("CSV format Telco Customer Churn", type=["csv"], key="upload_file",
                             help="Kolom wajib sama dengan dataset IBM Telco (tanpa kolom Churn juga diterima).")
            if st.session_state.get("upload_file") is None:
                st.caption("Belum ada file. Menampilkan dataset bawaan.")

        info = _resolve_dataset()
        st.session_state["dataset"] = info

        if info["errors"]:
            st.error("Dataset unggahan tidak dapat dipakai:\n\n" + "\n".join(f"- {e}" for e in info["errors"][:6])
                     + "\n\nMenampilkan dataset bawaan.")
        for w in info["warnings"][:5]:
            st.warning(w)
        if info["error_fatal"]:
            st.error(info["error_fatal"])
        elif info["table"] is not None:
            st.caption(info["label"])

        st.divider()
        st.markdown("**Model aktif**")
        try:
            b = load_model_bundle()
            st.markdown(
                f'<div class="side-card"><b>Churn:</b> {escape(str(b["model_name"]))}<br>'
                f'<b>Segmentasi:</b> {escape(str(b["final_method"]).upper())}, k = {int(b["k"])}<br>'
                f'<b>Threshold (F2):</b> {float(b["threshold"]):.2f}</div>', unsafe_allow_html=True)
            for n in b.get("_version_notes", []):
                st.warning(f"Versi library berbeda dari training: {n}")
        except ModelLoadError as exc:
            st.error(str(exc))
        st.caption("Model dilatih di Google Colab dan hanya dimuat, tidak dilatih ulang di aplikasi ini.")
    return info


def _called_from_app() -> bool:
    """True jika halaman sedang dijalankan lewat app.py (st.navigation); False jika dibuka langsung lewat URL/refresh."""
    target = os.path.abspath(str(config.ROOT / "app.py"))
    frame = sys._getframe(1)
    while frame is not None:
        if os.path.abspath(frame.f_code.co_filename) == target:
            return True
        frame = frame.f_back
    return False


def bootstrap() -> None:
    """Panggil di baris pertama setiap halaman.

    Lewat app.py: tidak melakukan apa-apa (app.py sudah memasang tema dan sidebar).
    Dibuka langsung (URL/refresh): memasang konfigurasi halaman, tema, dan sidebar sendiri.
    """
    if _called_from_app():
        return
    try:
        st.set_page_config(page_title="Telco Customer Intelligence", page_icon=":material/cell_tower:",
                           layout="wide", initial_sidebar_state="expanded")
    except Exception:  # noqa: BLE001  (sudah diset)
        pass
    inject_css()
    render_sidebar()


def get_dataset() -> tuple[pd.DataFrame, str, dict]:
    """Dipakai halaman: (tabel_analitik, label_sumber, info). Berhenti dengan pesan ramah jika data tidak tersedia."""
    info = st.session_state.get("dataset") or render_sidebar()
    if info.get("table") is None:
        st.error(info.get("error_fatal") or "Dataset tidak tersedia.")
        st.stop()
    return info["table"], info["label"], info


def stop_if_empty(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("Tidak ada pelanggan yang cocok dengan filter saat ini. Longgarkan filter di atas.")
        st.stop()
