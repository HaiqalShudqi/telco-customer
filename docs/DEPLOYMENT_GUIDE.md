# Panduan Deployment: Repo Baru di GitHub dan Streamlit Community Cloud

Panduan ini membuat **satu repository baru khusus aplikasi**, lalu mendeploy-nya gratis di Streamlit Community Cloud. Repo lama (`Telco-Customers`) tidak disentuh dan tidak dibutuhkan.

Yang Anda butuhkan hanya:

1. File **`telco-customer-intelligence_FINAL.zip`**.
2. Dua file dari proyek lokal Anda: `test_predictions.csv` dan `test_predictions_meta.json` (dari folder `reports\metrics`).
3. Akun GitHub dan GitHub Desktop.

---

## Langkah 1. Siapkan folder final

1. Klik kanan `telco-customer-intelligence_FINAL.zip` → **Extract All** → ke `C:\temp\app-final`.
2. Buka `C:\temp\app-final\telco-customer-intelligence`. Di dalamnya harus langsung terlihat `app.py`, `requirements.txt`, dan folder `pages`, `utils`, `models`, `data`, `reports`, `notebooks`. Folder inilah **folder final**.
3. Dari proyek lokal Anda, salin dua file dari `reports\metrics\`:
   - `test_predictions.csv`
   - `test_predictions_meta.json`

   lalu tempel ke `reports\metrics\` di folder final. (Dua file ini tidak ada di ZIP karena dibuat dari Colab Anda. Tanpa keduanya, aplikasi tetap jalan, hanya grafik di halaman Methodology menjadi statis.)
4. **Jangan** menyalin folder `.venv` atau apa pun dari proyek lokal selain dua file di atas.

## Langkah 2. Pasang dan masuk ke GitHub Desktop

1. Unduh dan pasang **GitHub Desktop** (desktop.github.com).
2. Buka, klik **Sign in to GitHub.com**, dan masuk dengan akun GitHub Anda (buat akun di github.com jika belum punya).

## Langkah 3. Buat repo baru dan unggah

1. Di GitHub Desktop: **File → New repository**.
2. Isi:
   - **Name:** `telco-customer-intelligence`
   - **Local path:** `C:\repos` (GitHub Desktop akan membuat folder `C:\repos\telco-customer-intelligence`)
   - **Initialize this repository with a README:** **jangan dicentang**
   - **Git ignore** dan **License:** biarkan *None*
3. Klik **Create repository**.
4. Buka `C:\repos\telco-customer-intelligence` di File Explorer (folder ini masih hampir kosong).
5. Buka folder final (Langkah 1). Tekan **Ctrl + A**, **Ctrl + C**, lalu tempel (**Ctrl + V**) ke dalam `C:\repos\telco-customer-intelligence`. Hasilnya, `app.py` berada **langsung** di folder itu, bukan di dalam subfolder.
6. Kembali ke GitHub Desktop. Tab **Changes** menampilkan daftar file baru. Periksa:
   - ada `app.py`, `requirements.txt`, `pages/`, dan `reports/metrics/test_predictions.csv`;
   - **tidak ada** `.venv` atau `__pycache__`.
7. Di kolom **Summary** (kiri bawah) ketik `Aplikasi Streamlit Customer Intelligence`. Klik **Commit to main**. (Jika diminta nama dan email Git, isi nama dan email akun GitHub Anda.)
8. Klik **Publish repository** (tombol di bagian atas). Di jendela yang muncul, **hilangkan centang "Keep this code private"** supaya repo publik, lalu klik **Publish Repository**.

## Langkah 4. Periksa di GitHub

Buka `https://github.com/NAMA-AKUN-ANDA/telco-customer-intelligence` (ganti dengan nama akun Anda). Pastikan:

- [ ] `app.py` dan `requirements.txt` ada di halaman depan (root).
- [ ] Ada folder `pages`, `utils`, `models`, `data`, `metadata`, `reports`, `notebooks`.
- [ ] `models/customer_intelligence_bundle.joblib` ada dengan ukuran sekitar 355 KB.
- [ ] `reports/metrics/test_predictions.csv` ada.
- [ ] `requirements.txt` memuat `scikit-learn==1.6.1`.

## Langkah 5. Deploy di Streamlit Community Cloud

1. Buka **share.streamlit.io**, klik **Continue with GitHub**, dan izinkan akses.
2. Klik **Create app** → **Deploy a public app from GitHub**.
3. Isi:
   - **Repository:** `NAMA-AKUN-ANDA/telco-customer-intelligence`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pilih nama yang mudah diingat.
4. Klik **Advanced settings** dan pilih **Python 3.12**.
   Ini penting. Library dipatok untuk Python 3.12 (`pandas==2.2.3`, `scikit-learn==1.6.1`). Jika Python salah, instalasi bisa gagal; cara termudah memperbaikinya adalah menghapus aplikasi dan membuatnya ulang.
5. Tidak ada *Secrets* yang perlu diisi.
6. Klik **Deploy**. Build pertama sekitar 3 sampai 10 menit.

Setelah aplikasi tampil, salin URL-nya ke baris "Aplikasi online" di `README.md` (edit langsung di GitHub, ikon pensil), lalu *Commit changes*.

## Langkah 6. Daftar periksa setelah deploy

| Halaman | Yang harus terlihat |
|---|---|
| Sidebar | Model aktif: XGBoost, KMEANS k = 3, Threshold 0.39, tanpa kotak merah |
| Executive Overview | 7,043 pelanggan, churn 26.5%, tenure 32.4 bln, tagihan $64.76, risiko tinggi 1,409 |
| Customer Segmentation | S0 3,651, S1 1,165, S2 2,227 pelanggan |
| Customer Explorer | Cari `7590-VHVEG`: probabilitas 80,3%, risiko High |
| Churn Prediction | Contoh pertama lalu *Prediksi churn*: 73,2%, High, segmen S0, P2 |
| Retention Strategy | P1 434, nilai tinggi 1,771, revenue at risk 1,667,798 |
| Methodology → D | Slider threshold dan kurva ROC/PR interaktif; pada 0,39: TP 324, FN 50, FP 365, TN 670 |
| Unggah CSV | Pilih *Unggah CSV sendiri*, lalu unggah file valid dan tidak valid |

Buka juga salah satu halaman lewat URL langsung (mis. `.../Customer_Segmentation`) lalu refresh. Tema harus tetap tampil.

---

## Memperbarui aplikasi di kemudian hari

Ubah file di `C:\repos\telco-customer-intelligence`, lalu di GitHub Desktop: tulis *Summary* → **Commit to main** → **Push origin**. Streamlit otomatis men-deploy ulang. Untuk memulai ulang tanpa perubahan kode: menu aplikasi (⋮) → **Reboot app**.

Aplikasi gratis akan *tidur* setelah beberapa waktu tanpa pengunjung. Saat dibuka lagi, klik *Wake up* dan tunggu sekitar satu menit.

---

## Pemecahan masalah

| Gejala | Penyebab dan solusi |
|---|---|
| Tombol **Publish repository** tidak ada | Anda belum membuat commit. Selesaikan langkah 3 butir 7 dulu. |
| GitHub Desktop menampilkan ribuan file di Changes | Anda tidak sengaja menyalin `.venv`. Hapus folder itu dari `C:\repos\telco-customer-intelligence`. |
| Repo di GitHub tidak punya folder `pages` atau `app.py` di root | File tertempel di dalam subfolder (mis. `telco-customer-intelligence\telco-customer-intelligence`). Pindahkan isinya satu tingkat ke atas, lalu commit dan push lagi. |
| Build gagal saat memasang `pandas` atau `scikit-learn` | Python bukan 3.12. Hapus aplikasi dan buat ulang dengan Python 3.12. |
| `ModuleNotFoundError: utils` | *Main file path* salah. Harus `app.py` dan berada di root repo. |
| Kotak merah "Model gagal dimuat" | Versi library tidak sama dengan saat training. Periksa `requirements.txt` dan file `models/customer_intelligence_bundle.joblib` di GitHub. Aplikasi otomatis mencoba bundle cadangan. |
| `FileNotFoundError` untuk data atau reports | Ada folder yang belum ter-push. Nama file peka huruf besar/kecil di Linux. |
| Methodology hanya menampilkan gambar statis | `test_predictions.csv` dan `test_predictions_meta.json` belum ada di `reports/metrics/` di GitHub, atau tidak lolos validasi (peringatan kuning menjelaskan alasannya). |
| Tidak bisa memilih repo di share.streamlit.io | Beri Streamlit akses ke repo di GitHub: *Settings → Applications → Streamlit*. |
| Aplikasi lambat saat pertama dibuka | Normal: model dimuat sekali per proses server, lalu di-cache. |
