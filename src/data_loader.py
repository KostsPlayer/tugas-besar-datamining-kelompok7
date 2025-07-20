# src/data_loader.py

"""
data_loader.py

Module ini digunakan untuk memuat dataset gabungan antara historis dan sentimen dari direktori `data/processed`.
"""

import pandas as pd
import os

# Folder path relatif dari root repository
PATH = "data/processed/"

def load_csv(filename):
    """
    Memuat file CSV dari folder data.
    
    Parameters:
        filename (str): Nama file (contoh: 'data.csv')
    
    Returns:
        pd.DataFrame: Dataframe dari file yang dimuat
    """
    file_path = os.path.join(PATH, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    
    return pd.read_csv(file_path)


def preview_data(df):
    """
    Menampilkan preview awal dari dataframe
    
    Parameters:
        df (pd.DataFrame): Dataframe yang akan dipreview
        rows (int): Jumlah baris untuk ditampilkan
    """
    df
    df.info()


# Contoh pemanggilan (hapus saat produksi):
if __name__ == "__main__":
    try:
        df = load_csv("your_dataset.csv")  # ganti nama file sesuai kebutuhan
    except Exception as e:
        print(e)
