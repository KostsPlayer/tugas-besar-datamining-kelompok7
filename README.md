# 📊 Skeleton Proyek Data Mining

Repositori ini berisi kerangka kerja (skeleton) proyek data mining yang terstruktur dan modular.

---

## 🗂 Struktur Direktori

```
tube_data_mining/
│
├── data/                      # Folder untuk menyimpan dataset
│   ├── raw/                   # Data mentah (belum diproses)
│   └── processed/             # Data setelah preprocessing
│
├── notebook/                 # Jupyter Notebook interaktif
│   ├── eda_template.ipynb     # Template untuk eksplorasi data
│   ├── preprocessing_template.ipynb  # Template untuk preprocessing
│   └── modeling_template.ipynb       # Template untuk pelatihan model
│
├── report/                   # Template laporan akhir
│   ├── laporan-akhir_template.pdf
│   ├── lampiran_template.docx
│   └── struktur-lampiran.md
│
├── src/                      # Source code modular
│   ├── data_loader.py         # Fungsi load dan simpan data
│   ├── model.py               # Fungsi training model
│   ├── utils.py               # Evaluasi model dan fungsi bantu
│   ├── main.py                # Main pipeline untuk dijalankan via terminal
│   └── main_notebook.ipynb    # Versi notebook dari main.py
│
├── run.sh                    # Script bash untuk menjalankan pipeline
├── requirements.txt          # Daftar dependensi Python
└── README.md                 # Dokumentasi ini
```

---

## 🚀 Cara Menjalankan

### ✅ 1. Persiapkan Environment

Install dependensi:
```bash
pip install -r requirements.txt
```

### ✅ 2. Jalankan Pipeline

#### 💻 Via Terminal:
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

- **`data_loader.py`**: fungsi `load_raw_data()` dan `save_processed_data()`
- **`model.py`**: fungsi `train_model()`, split data, dan prediksi
- **`utils.py`**: evaluasi model (akurasi, classification report, dll.)

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
