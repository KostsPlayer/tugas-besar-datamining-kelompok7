# src/historis.py

"""
historis.py

Module ini digunakan untuk memanggil model-model yang hanya menggunakan fitur historis.
"""

from model import models, evaluate_model
import pandas as pd


def historical_models(X_hist, y):
    """
    Menjalankan pemodelan dan evaluasinya terhadap fitur historis yang telah ditentukan.

    Parameter:
    - df: DataFrame utama yang berisi semua data fitur
    - y: target variabel (label)
    - X: DataFrame fitur untuk training model akhir

    Fungsi ini melakukan:
    1. Evaluasi setiap algoritma pada fitur historis dengan K-Fold TimeSeries sebanyak 5 kali
    2. Training akhir model untuk evaluasi akhir (confusion matrix, report)

    Output:
    - results_hist: dictionary berisi metrik evaluasi tiap model
    """
    
    print("🔹 Tahap 1: Data Historis")
    results_hist = {}

    for name, model in models.items():
        print(f"Evaluating {name} with historical data")
        results_hist[name] = evaluate_model(model, X_hist, y)

    return results_hist


def summary_hist(results_hist):
    """
    Menampilkan seluruh hasil evaluasi model historis dalam rata-rata dan standar deviasi yang berbentuk dataframe.

    Parameter:
    - results_hist: dictionary berisi metrik evaluasi tiap model
    """
    
    print("\n Rangkuman Evaluasi Model (Historis):")
    df_historis = pd.DataFrame(results_hist).T

    return df_historis


"""
# Contoh pemanggilan (hapus saat produksi)
if __name__ == "__main__":
    try:
        df = "ini adalah contoh dataframe"                      # Ganti dengan dataframe yang sesuai
        features_hist = "ini adalah contoh fitur historis"      # Ganti dengan fitur historis yang sesuai
        y = "ini adalah contoh target"                          # Ganti dengan target yang sesuai

        results_hist = historical_models(df, features_hist, y)  # lakukan pemodelan dan evaluiasi pada data historis
        summary_hist(results_hist)                              # tampilkan rangkuman hasil evaluasi
    except Exception as e:
        print(e)
"""