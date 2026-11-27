"""Aturan tindakan bisnis berbasis prioritas. Ini panduan berbasis aturan, bukan keluaran model."""
from __future__ import annotations

PRIORITY_ACTIONS = {
    "P1 - Selamatkan segera": (
        "Risiko tinggi dan nilai tinggi. Hubungi dalam waktu dekat dengan penawaran retensi yang dipersonalisasi, misalnya "
        "insentif pindah ke kontrak 1-2 tahun, penyesuaian paket, atau bantuan dukungan teknis."),
    "P2 - Retensi terstruktur": (
        "Risiko tinggi dengan nilai standar, atau risiko menengah dengan nilai tinggi. Masukkan ke kampanye retensi terstruktur "
        "(komunikasi terjadwal dan penawaran standar), lalu ukur hasilnya dengan uji A/B."),
    "P3 - Pantau": (
        "Risiko menengah dengan nilai standar. Pantau skor secara berkala dan gunakan komunikasi ringan; hindari insentif besar "
        "yang belum tentu diperlukan."),
    "P4 - Pertahankan": (
        "Risiko rendah. Pertahankan lewat program loyalitas dan tawarkan layanan tambahan secara selektif."),
}

PRIORITY_DEFINITION = {
    "P1 - Selamatkan segera": "Risiko High dan nilai High",
    "P2 - Retensi terstruktur": "Risiko High + nilai Standard, atau risiko Medium + nilai High",
    "P3 - Pantau": "Risiko Medium dan nilai Standard",
    "P4 - Pertahankan": "Risiko Low (semua nilai)",
}

REVENUE_AT_RISK_NOTE = (
    "<b>Revenue at risk</b> = probabilitas churn terkalibrasi × proksi nilai tahunan (tagihan bulanan × 12). "
    "Ini <b>indeks prioritisasi relatif</b>, bukan prakiraan kerugian finansial: data tidak memuat margin, biaya, atau nilai umur "
    "pelanggan (CLV), dan probabilitas model bukan kepastian. Gunakan untuk mengurutkan, bukan untuk menyusun angka anggaran."
)
