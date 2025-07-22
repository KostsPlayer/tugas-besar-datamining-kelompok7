# 📊 Skeleton Proyek Data Mining

Repositori ini berisi kerangka kerja (skeleton) proyek data mining yang terstruktur dan modular.

---

## Stock Price Sentiment
Proyek ini menilai apakah informasi sosial real-time dapat meningkatkan akurasi prediksi dibanding hanya menggunakan data historis harga saham

## Anggota Kelompok
- Muhammad Azka Nuril Islami `(714220001)`
- Gaizka Wisnu Prawira `(714220011)`
- Muhammad Fathir `(714220021)`
- Salwa Mutfia Indah Putri `(714220026)`

## Sumber Dataset
- `Historis` (website investing.com)
- `Sentimen` (X atau Twitter)
- `Gabungan` (Dataset gabungan antara data historis dan data sentimen)

## Algoritma
- Random Forest
- XGBoost
- Logistik Regresi
- SVC
- MLP

## Evaluasi Didasarkan pada F1 Score Mean – Std (Standar Deviasi)
| Model      | Random Forest | XGBoost     | Logistik Regresi | SVC         | MLP         |
|------------|---------------|-------------|------------------|-------------|-------------|
| Historis   | 0.44 – 0.32   | 0.26 – 0.20 | 0.19 – 0.27      | 0.44 – 0.31 | 0.44 – 0.32 |
| Merged     | 0.51 – 0.37   | 0.55 – 0.39 | 0.45 – 0.35      | 0.22 – 0.31 | 0.42 – 0.35 |
| Optimation | 0.51 – 0.37   | 0.55 – 0.39 | 0.45 – 0.35      | 0.56 – 0.42 | 0.47 – 0.41 |   

## 🗂 Struktur Direktori

```
tube_data_mining/
│
├── data/                              # Folder untuk menyimpan dataset
│   ├── raw/                           # Data mentah (belum diproses)
│   └── processed/                     # Data setelah preprocessing
│
├── notebook/                          # Jupyter Notebook interaktif
│   ├── eda_template.ipynb             # Template untuk eksplorasi data
│   └── preprocessing_template.ipynb   # Template untuk preprocessing
│
├── report/                            # Template laporan akhir
│   ├── laporan-akhir.docx
│   ├── laporan-akhir.pdf
│   └── presentasi-akhir.pdf
│
├── src/                               # Source code modular
│   ├── data_loader.py                 # Fungsi load dan simpan data
|   ├── preprocessing.py               # Fungsi preprocessing data dan pengadaan fitur-fitur pendukung baru
│   ├── model.py                       # Fungsi template model
|   ├── historis.py                    # Fungsi model berdasarkan sekedar data dan fitur historis
|   ├── merged.py                      # Fungsi model berdasarkan data dan fitur gabungan antara historis dan sentimen
|   ├── optimalization.py              # Fungsi mengoptimalkan model gabungan berdasarkan pendekatan parameter terbaik
|   ├── visualization.py               # Fungsi visualisasi dari hasil perbandingan semua model
│   ├── main.py                        # Main pipeline untuk dijalankan via terminal
│   └── main_notebook.ipynb            # Versi notebook dari main.py
│
├── run.sh                             # Script bash untuk menjalankan pipeline
├── requirements.txt                   # Daftar dependensi Python
└── README.md                          # Dokumentasi ini
```

---

## 🚀 Cara Menjalankan Pipeline

#### 💻 Via Terminal (Git):
```bash
bash run.sh
```

#### 📒 Via Jupyter Notebook:
Buka dan jalankan:
```text
src/main_notebook.ipynb
```

---

## 📦 Struktur Modular

- **`data_loader.py`**: fungsi `load_csv()` dan `preview_data()`
- **`preprocessing.py`**: fungsi `preprocessing()` dan `feature_engineering()`
- **`model.py`**: variabel `models`, fungsi `directional_accuracy()` dan `evaluate_model()`
- **`historis.py`**: fungsi `historical_models()` dan `summary_hist()`
- **`merged.py`**: fungsi `merged_models()` dan `summary_merged()`
- **`optimalization.py`**: fungsi `optimize_func()`, `all_optimization()`, `optimalization_models()`, dan `summary_optimalization()`
- **`visualization.py`**: fungsi `process_historis()`, `process_merged()`, `process_optimation()`, `marker_style()`, dan `plot_results()`

---

## 📓 Catatan

- Semua path diasumsikan relatif dari root project
- Tambahkan file data kamu ke dalam `data/raw/`
- Hasil preprocessing disimpan di `data/processed/` 
- Pastikan target label diberi nama kolom `target` (atau sesuaikan di script)

---

## 👩‍💻 Kontributor

- Seluruh Anggota Kelompok

---

## 📄 Lisensi

Proyek ini bersifat open-source dan bebas digunakan untuk edukasi dan pengembangan pribadi.
