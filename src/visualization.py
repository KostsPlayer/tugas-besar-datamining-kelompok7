# src/visualization.py

"""
visualization.py

Module ini digunakan untuk melakukan visualisasi agar hasil dari model dapat dilihat dan dipahami oleh orang secara umum.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def process_historis(df_historis):
    """
    Proses visualisasi optimasi model.

    Parameters:
        df_historis (pd.DataFrame): Dataframe yang berisi hasil model historis
    
    Returns:
        DataFrame: Dataframe yang sudah diproses untuk visualisasi
    """

    df_historis_pro = df_historis.copy()

    df_historis_pro = df_historis_pro.reset_index()
    df_historis_pro.rename(columns={'index': 'model'}, inplace=True)

    # Ekstrak nama algoritma: RandomForest, XGBoost, LogReg, SVC, MLP
    df_historis_pro['model'] = df_historis_pro['model'].str.extract(r'^([A-Za-z]+)')
    
    return df_historis_pro


def process_merged(df_merged):
    """
    Proses visualisasi optimasi model.

    Parameters:
        df_merged (pd.DataFrame): Dataframe yang berisi hasil model merged
    
    Returns:
        DataFrame: Dataframe yang sudah diproses untuk visualisasi
    """

    df_merged_pro = df_merged.copy()

    df_merged_pro = df_merged_pro.reset_index()
    df_merged_pro.rename(columns={'index': 'model'}, inplace=True)

    # Ekstrak nama algoritma: RandomForest, XGBoost, LogReg, SVC, MLP
    df_merged_pro['model'] = df_merged_pro['model'].str.extract(r'^([A-Za-z]+)')
    
    return df_merged_pro


def process_optimation(df_optimized):
    """
    Proses visualisasi optimasi model.

    Parameters:
        df_optimized (pd.DataFrame): Dataframe yang berisi hasil optimasi model
    
    Returns:
        DataFrame: Dataframe yang sudah diproses untuk visualisasi
    """

    df_optimized_pro = df_optimized.copy()

    df_optimized_pro = df_optimized_pro.reset_index()
    df_optimized_pro.rename(columns={'index': 'model_name'}, inplace=True)

    # Ekstrak nama algoritma: RandomForest, XGBoost, LogReg, SVC, MLP
    df_optimized_pro['model'] = df_optimized_pro['model_name'].str.extract(r'^([A-Za-z]+)')

    # Pastikan kolom f1_mean bertipe float (hindari NaN/string)
    df_optimized_pro['f1_mean'] = pd.to_numeric(df_optimized_pro['f1_mean'], errors='coerce')

    # Ambil model terbaik (berdasarkan f1 tertinggi) untuk setiap algoritma
    best_models = df_optimized_pro.loc[df_optimized_pro.groupby('model')['f1_mean'].idxmax()]

    return best_models


def marker_style(df_historis, df_combined, best_models):
    """
    Menambahkan kolom untuk penanda dataframe (eksperimen).

    Parameters:
        df_historis (pd.DataFrame): Dataframe hasil model historis terbaru
        df_combined (pd.DataFrame): Dataframe hasil model gabungan terbaru
        best_models (pd.DataFrame): Dataframe hasil model optimasi terbaru

    Returns:
        pd.DataFrame: Dataframe yang sudah ditambahkan kolom penanda
    """
    
    # Tambahkan kolom untuk penanda dataframe (eksperimen)
    df_historis['source'] = 'Model Historis'
    df_combined['source'] = 'Model Gabungan'
    best_models['source'] = 'Model Optimasi'

    # Gabungkan semua dataframe
    df_all = pd.concat([df_historis, df_combined, best_models], ignore_index=True)

    order = ['Model Historis', 'Model Gabungan', 'Model Optimasi']
    df_all['source'] = pd.Categorical(df_all['source'], categories=order, ordered=True)

    return df_all


def plot_results(df_all):
    """
    Plot hasil dari semua model.

    Parameters:
        df_all (pd.DataFrame): Dataframe yang berisi hasil dari semua model
    """
    
    sns.scatterplot(data=df_all, x='f1_mean', y='f1_std', hue='source')
    plt.title("Trade-off antara F1 Mean dan Std")
    plt.xlabel("F1 Mean")
    plt.ylabel("F1 Std")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gabungkan mean dan std sebagai string anotasi
    df_all['f1_label'] = df_all.apply(lambda x: f"{x['f1_mean']:.2f} – {x['f1_std']:.2f}", axis=1)

    # Pivot untuk heatmap dengan anotasi gabungan
    pivot_val = df_all.pivot(index='source', columns='model', values='f1_mean')
    pivot_label = df_all.pivot(index='source', columns='model', values='f1_label')

    # Visualisasi heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_val, annot=pivot_label, cmap='YlGnBu', fmt="", cbar_kws={'label': 'F1 Score'})
    plt.title("F1 Score Heatmap (Mean – Std)")
    plt.xlabel("Algoritma")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()