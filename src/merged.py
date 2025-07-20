# src/merged.py

"""
merged.py

Module ini digunakan untuk memanggil model-model yang menggunakan semua fitur.
"""

from model import models, evaluate_model
import pandas as pd


def merged_models(df, features_merged, y):
    """
    Menjalankan pemodelan dan evaluasinya terhadap semua fitur yang telah ditentukan.

    Parameter:
    - df: DataFrame utama yang berisi semua data fitur
    - y: target variabel (label)
    - X: DataFrame fitur untuk training model akhir

    Fungsi ini melakukan:
    1. Evaluasi setiap algoritma pada semua fitur dengan K-Fold TimeSeries sebanyak 5 kali
    2. Training akhir model untuk evaluasi akhir (confusion matrix, report)

    Output:
    - results_merged: dictionary berisi metrik evaluasi tiap model
    """
    X_merged = df[features_merged]

    print("🔹 Tahap 1: Data Historis")
    results_merged = {}

    for name, model in models.items():
        print(f"Evaluating {name} with merged data")
        results_merged[name] = evaluate_model(model, X_merged, y)

    return results_merged


def summary_merged(results_merged):
    """
    Menampilkan seluruh hasil evaluasi model merged dalam rata-rata dan standar deviasi yang berbentuk dataframe.

    Parameter:
    - results_merged: dictionary berisi metrik evaluasi tiap model
    """
    
    print("\n Rangkuman Evaluasi Model (Merged):")
    df_merged = pd.DataFrame(results_merged).T
    return df_merged


# Contoh pemanggilan (hapus saat produksi)
if __name__ == "__main__":
    try:
        df = "ini adalah contoh dataframe"                      # Ganti dengan dataframe yang sesuai
        features_merged = "ini adalah contoh fitur merged"      # Ganti dengan fitur merged yang sesuai
        y = "ini adalah contoh target"                          # Ganti dengan target yang sesuai

        results_merged = merged_models(df, features_merged, y)  # lakukan pemodelan dan evaluiasi pada data merged
        summary_merged(results_merged)                          # tampilkan rangkuman hasil evaluasi
    except Exception as e:
        print(e)
