# src/main.py

"""
Main pipeline for Data Mining project.
Steps:
1. Load dataset
2. Preprocessing
3. Train & evaluate model
4. Optimalization model
"""

from data_loader import load_csv, preview_data
from preprocessing import preprocessing, feature_engineering
from historis import historical_models, summary_hist
from merged import merged_models, summary_merged
from optimalization import all_optimization, optimalization_models, summary_optimalization
from visualization import process_historis, process_merged, process_optimation, marker_style, plot_results

import warnings
warnings.filterwarnings("ignore")


def main():
    # 1. Load dataset
    df_default = load_csv("dataset_model.csv")  # Ganti sesuai nama file dataset
    
    if df_default is None:
        print("Dataset tidak ditemukan.")
        return

    # Preview data
    preview_data(df_default)

    # 2. Preprocessing
    preprocessing(df_default)
    feature_engineering(df_default)

    df = df_default.copy()

    # cek apakah kolom target ada
    if "target" not in df.columns:
        print("Kolom 'target' tidak ditemukan dalam dataset.")
        return

    y = df["target"]

    # 3. Train & evaluate model

    # historis
    features_hist = ['Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%']
    X_hist = df[features_hist]
    results_hist = historical_models(X_hist, y)
    df_historis = summary_hist(results_hist)

    # merged (historis + sentimen)
    features_merged = [
    'Terakhir', 'Pembukaan', 'Tertinggi', 'Terendah', 'Vol.', 'Perubahan%',
    'avg_signed_sentiment', 'count_positive', 'count_negative', 'count_neutral', 'total_tweets',
    'range', 'day_return', 'sentiment_ratio', 'tweet_intensity',
    'lag_1', 'lag_2'
    ]
    X_merged = df[features_merged]
    results_merged = merged_models(X_merged, y)
    df_merged = summary_merged(results_merged)

    # 4. Optimalization model
    optimized_models = all_optimization(X_merged, y)
    results_optimized = optimalization_models(optimized_models, X_merged, y) 
    df_optimized = summary_optimalization(results_optimized)


    # 5. Visualisasi
    df_historis_pro = process_historis(df_historis)
    df_combined_pro = process_merged(df_merged)
    best_models = process_optimation(df_optimized)

    df_all = marker_style(df_historis_pro, df_combined_pro, best_models)
    plot_results(df_all)


if __name__ == "__main__":
    main()
