# Telco Customer Intelligence: Ringkasan Temuan (dibangkitkan otomatis dari hasil eksekusi)

## 1. Temuan utama EDA
- 7,043 pelanggan; churn rate keseluruhan **26.54%** (rasio kelas 2.77:1).
- Kualitas data: 11 nilai TotalCharges kosong (semua tenure = 0, diisi 0); tidak ada duplikasi penuh, outlier IQR 0, tidak ada pelanggaran logika bisnis.
- Variabel dengan gap churn terbesar antar-kategori: Contract, tenure_group, InternetService, OnlineSecurity, TechSupport.

## 2. Temuan statistik
- 15 dari 17 variabel kategorikal berasosiasi signifikan dengan churn setelah koreksi Holm (alpha 0,05).
- Asosiasi terkuat (Cramér's V): Contract (0.410), tenure_group (0.348), OnlineSecurity (0.347), TechSupport (0.343), InternetService (0.322).
- Numerik (Mann-Whitney): rank-biserial r tenure = -0.480, MonthlyCharges = 0.242, TotalCharges = -0.301.
- Semua temuan adalah **asosiasi, bukan kausalitas**.

## 3-4. Segmentasi pelanggan
- Metode final: **KMEANS**, k = **3** (Silhouette K-Means = 0.403; stabilitas ARI K-Means = 0.987).
- Hubungan segmen-churn (pasca-hoc): Cramér's V = 0.381 (p = 1.25e-223).
- **S0: Menengah | tagihan sedang | Kontrak bulanan** (n=3,651, 51.8%): tenure rata-rata 14.7 bln, tagihan 63.0, kontrak bulanan 95%, churn 42.4% (CI95 40.8-44.0)
- **S1: Menengah | tagihan rendah | Kontrak 2 thn** (n=1,165, 16.5%): tenure rata-rata 42.0 bln, tagihan 24.5, kontrak bulanan 1%, churn 2.0% (CI95 1.3-2.9)
- **S2: Loyal | tagihan tinggi | Kontrak 2 thn** (n=2,227, 31.6%): tenure rata-rata 56.4 bln, tagihan 88.7, kontrak bulanan 18%, churn 13.4% (CI95 12.1-14.9)

## 5. Churn analysis
- Risiko relatif tertinggi: Kontrak bulanan (RR 6.32), Tanpa OnlineSecurity (pelanggan internet) (RR 3.63), Tanpa TechSupport (pelanggan internet) (RR 3.51).

## 6. Model prediksi terbaik
- Model terpilih (CV PR-AUC tertinggi): **XGBoost** (CV PR-AUC 0.668, CV ROC-AUC 0.849).
- Test set @0,5: recall 0.807, precision 0.511, F1 0.626, ROC-AUC 0.847, PR-AUC 0.667.
- Test set @threshold 0.39 (F2): recall 0.866, precision 0.470, F1 0.610.
- Bootstrap CI95 ROC-AUC 0.828-0.870; PR-AUC 0.624-0.718.

## 7. Faktor penting dalam prediksi
- Permutation importance teratas: tenure, Contract, InternetService, TotalCharges, MonthlyCharges.
- SHAP teratas: Contract, tenure, InternetService, MonthlyCharges, PaymentMethod.
- Importance menunjukkan kontribusi pada prediksi model, **bukan** hubungan sebab-akibat.

## 8. Strategi retensi (hipotesis untuk diuji A/B)
- **S0: Menengah | tagihan sedang | Kontrak bulanan** [risiko TINGGI]: Insentif migrasi ke kontrak 1-2 tahun (diskon/benefit di awal) | Dorong pembayaran otomatis (transfer bank/kartu kredit) dengan insentif kecil | Uji coba bundling add-on (keamanan/dukungan teknis) gratis atau berdiskon _(bukti: churn 42.4% vs 26.5% rata-rata; 95% kontrak bulanan vs 55%; 46% electronic check vs 34%; rata-rata add-on 1.3 vs 2.0)_
- **S1: Menengah | tagihan rendah | Kontrak 2 thn** [risiko RENDAH]: Pantau; tidak ada penyimpangan profil yang kuat dari rata-rata _(bukti: churn 2.0% vs 26.5% rata-rata)_
- **S2: Loyal | tagihan tinggi | Kontrak 2 thn** [risiko RENDAH]: Pertahankan: program loyalitas, upsell layanan tambahan secara selektif _(bukti: churn 13.4% vs 26.5% rata-rata; tenure rata-rata 56 bln)_

## 9. Keterbatasan dataset
- Data satu snapshot tanpa tanggal: tidak ada urutan waktu, musiman, atau definisi jelas kapan churn terjadi; analisis survival mengandalkan asumsi sensor.
- Tidak ada data biaya, margin, riwayat keluhan, kualitas jaringan, atau penawaran kompetitor; `annual_value_proxy` hanya proksi pendapatan, bukan CLV.
- Satu operator dan satu wilayah yang tidak dijelaskan; generalisasi terbatas.
- MonthlyCharges adalah tagihan saat ini, sedangkan TotalCharges adalah akumulasi historis (redundan dengan tenure x MonthlyCharges).
- Probabilitas model berbobot kelas perlu kalibrasi; kalibrasi isotonik dilakukan pada skor OOF dan tetap perlu dipantau.
- Efek intervensi retensi tidak dapat diukur dari data observasional.

## 10. Pengembangan selanjutnya
- Uji A/B atau uplift modeling untuk mengukur efek nyata tiap tawaran retensi.
- Data longitudinal (tagihan bulanan, tiket layanan, penggunaan) dan model survival/hazard.
- Model biaya eksplisit (biaya penawaran vs nilai pelanggan) untuk memilih threshold optimal.
- Pemantauan drift dan retraining berkala; evaluasi keadilan antar kelompok demografi.
