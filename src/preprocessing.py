# src/preprocessing.py

"""
preprocessing.py

Module ini digunakan untuk membersihkan dataset hingga siap digunakan untuk pemodelan serta menambahkan fitur-fitur baru dari perkembangan dataset dengan menyesuiakan dengan kebutuhan model.
"""

import pandas as pd


def preprocessing(df):
    """
    Melakukan preprocessing pada dataframe yang dimuat.
    
    Parameters:
        df (pd.DataFrame): Dataframe yang akan diproses
    
    Returns:
        pd.DataFrame: Dataframe yang sudah dibersihkan dan siap digunakan
    """
    df.dropna(inplace=True)

    # Drop kolom 'date' jika ada
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)

    # Format tanggal
    if 'Tanggal' in df.columns:
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df.sort_values(by='Tanggal', inplace=True)

    # Fungsi bantu untuk parsing volume
    def parse_volume(vol_str):
        if isinstance(vol_str, str):
            vol_str = vol_str.replace(',', '.')
            if vol_str.endswith('M'):
                return float(vol_str[:-1]) * 1_000_000
            elif vol_str.endswith('K'):
                return float(vol_str[:-1]) * 1_000
            else:
                return float(vol_str)
        return vol_str

    if 'Vol.' in df.columns:
        df['Vol.'] = df['Vol.'].apply(parse_volume)

    if 'Perubahan%' in df.columns:
        df['Perubahan%'] = df['Perubahan%'].str.replace('%', '', regex=False)
        df['Perubahan%'] = df['Perubahan%'].str.replace(',', '.', regex=False).astype(float)

    return df


def feature_engineering(df):
    """
    Menambahkan fitur-fitur tambahan, fitur lag, dan kolom target ke dataframe.

    Parameters:
        df (pd.DataFrame): Dataframe hasil preprocessing

    Returns:
        pd.DataFrame: Dataframe dengan fitur baru dan target
    """
    # Fitur tambahan
    df['range'] = df['Tertinggi'] - df['Terendah']
    df['day_return'] = df['Terakhir'].pct_change()
    df['sentiment_ratio'] = df['count_positive'] / (df['count_negative'] + 1)
    df['tweet_intensity'] = df['total_tweets'] / (df['Vol.'] + 1)

    # Lag features
    df['lag_1'] = df['Terakhir'].shift(1)
    df['lag_2'] = df['Terakhir'].shift(2)

    # Target: apakah harga besok naik dari hari ini?
    df['target'] = (df['Terakhir'].shift(-1) > df['Terakhir']).astype(int)

    # Keterangan target
    df['keterangan_target'] = df['target'].map({1: 'Naik', 0: 'Turun/Stagnan'})

    # Drop baris yang mengandung NaN setelah pembuatan fitur
    df.dropna(inplace=True)

    return df


# Contoh pemanggilan (hapus saat produksi)
if __name__ == "__main__":
    try:
        df = "ini adalah contoh dataframe"      # Ganti dengan dataframe yang sesuai
        df = preprocessing(df)                  # lakukan preprocessing
        df = feature_engineering(df)            # lakukan feature engineering
    except Exception as e:
        print(e)
