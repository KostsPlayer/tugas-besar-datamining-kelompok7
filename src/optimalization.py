# src/optimalization.py

"""
optimalization.py

Module ini digunakan untuk memanggil model-model yang menggunakan semua fitur dan telah dioptimalkan.
"""

from model import evaluate_model

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def optimize_func(model, param_grid, X, y, n_splits=5):
    """
    Melakukan hyperparameter tuning menggunakan GridSearchCV dengan TimeSeriesSplit.

    Parameter:
    - model: model machine learning yang akan dioptimasi
    - param_grid: dictionary parameter grid untuk tuning
    - X, y: fitur dan label
    - n_splits: jumlah split untuk TimeSeriesSplit

    Output:
    - best_estimator_: model terbaik hasil tuning
    - best_params_: parameter terbaik
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=tscv, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def all_optimization(X_merged, y):
    """
    Menjalankan proses optimasi hyperparameter untuk semua model dengan menggunakan TimeSeriesSplit (3, 5, dan 10 fold).

    Model yang dioptimasi:
    - RandomForest
    - XGBoost
    - Logistic Regression (LogReg)
    - SVC
    - MLP

    Output:
    - optimized_models: dictionary berisi model terbaik & parameter per jumlah fold
    """

    optimized_models = {}
    optimized_folds = [3, 5, 10]

    # Random Forest
    rf_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
    }
    optimized_models['RandomForest'] = {}
    for n_fold in optimized_folds:
        print(f"\n🔹 Optimasi Random Forest dengan {n_fold}-fold TimeSeriesSplit")
        best_rf, best_rf_params = optimize_func(
            RandomForestClassifier(random_state=42),
            rf_grid, X_merged, y, n_splits=n_fold
        )
        print(f"✅ Best Params ({n_fold} fold):", best_rf_params)
        optimized_models['RandomForest'][f'{n_fold}_fold'] = {
            'model': best_rf,
            'params': best_rf_params
        }

    # XGBoost
    xgb_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 6],
        'model__learning_rate': [0.05, 0.1]
    }
    optimized_models['XGBoost'] = {}
    for n_fold in optimized_folds:
        print(f"\n🔹 Optimasi XGBoost dengan {n_fold}-fold TimeSeriesSplit")
        best_xgb, best_xgb_params = optimize_func(
            XGBClassifier(eval_metric='logloss'),
            xgb_grid, X_merged, y, n_splits=n_fold
        )
        print(f"✅ Best Params ({n_fold} fold):", best_xgb_params)
        optimized_models['XGBoost'][f'{n_fold}_fold'] = {
            'model': best_xgb,
            'params': best_xgb_params
        }

    # Logistic Regression
    logreg_grid = {
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l2'],
        'model__solver': ['lbfgs']
    }
    optimized_models['LogReg'] = {}
    for n_fold in optimized_folds:
        print(f"\n🔹 Optimasi LogReg dengan {n_fold}-fold TimeSeriesSplit")
        best_logreg, best_logreg_params = optimize_func(
            LogisticRegression(max_iter=1000),
            logreg_grid, X_merged, y, n_splits=n_fold
        )
        print(f"✅ Best Params ({n_fold} fold):", best_logreg_params)
        optimized_models['LogReg'][f'{n_fold}_fold'] = {
            'model': best_logreg,
            'params': best_logreg_params
        }

    # SVC
    svc_grid = {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf'],
        'model__gamma': ['scale', 'auto']
    }
    optimized_models['SVC'] = {}
    for n_fold in optimized_folds:
        print(f"\n🔹 Optimasi SVC dengan {n_fold}-fold TimeSeriesSplit")
        best_svc, best_svc_params = optimize_func(
            SVC(),
            svc_grid, X_merged, y, n_splits=n_fold
        )
        print(f"✅ Best Params ({n_fold} fold):", best_svc_params)
        optimized_models['SVC'][f'{n_fold}_fold'] = {
            'model': best_svc,
            'params': best_svc_params
        }

    # MLP
    mlp_grid = {
        'model__hidden_layer_sizes': [(100,), (50, 50)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001]
    }
    optimized_models['MLP'] = {}
    for n_fold in optimized_folds:
        print(f"\n🔹 Optimasi MLP dengan {n_fold}-fold TimeSeriesSplit")
        best_mlp, best_mlp_params = optimize_func(
            MLPClassifier(max_iter=1000, random_state=42),
            mlp_grid, X_merged, y, n_splits=n_fold
        )
        print(f"✅ Best Params ({n_fold} fold):", best_mlp_params)
        optimized_models['MLP'][f'{n_fold}_fold'] = {
            'model': best_mlp,
            'params': best_mlp_params
        }

    return optimized_models


def optimalization_models(optimized_models, X_merged, y):
    """
    Menjalankan optimalisasi terhadapt semua model dan evaluasinya.

    Parameter:
    - optimized_models: dictionary berisi model terbaik & parameter per jumlah fold
    - X_merged: DataFrame fitur untuk training model akhir
    - y: target variabel (label)

    Fungsi ini melakukan:
    1. Evaluasi kembali setiap algoritma untuk optimalisasi pada semua fitur dengan K-Fold TimeSeries sebanyak 3, 5, dan 10 kali
    2. Training akhir model untuk evaluasi akhir kembali (confusion matrix, report)

    Output:
    - results_optimized: dictionary berisi metrik evaluasi tiap model
    """

    print("\n Evaluasi Kembali Setelah Optimalisasi")
    results_optimized = {}

    for name, folds_dict in optimized_models.items():
        for fold_name, content in folds_dict.items():
            model = content['model']
            key = f"{name}_{fold_name}"
            results_optimized[key] = evaluate_model(model, X_merged, y, model_name=key)

    return results_optimized


def summary_optimalization(results_optimized):
    """
    Menampilkan seluruh hasil evaluasi model merged dalam rata-rata dan standar deviasi yang berbentuk dataframe.

    Parameter:
    - results_optimized: dictionary berisi metrik evaluasi tiap model
    """
    
    print("\n Rangkuman Evaluasi Model (Optimasi):")
    df_optimized = pd.DataFrame(results_optimized).T
    return df_optimized


# Contoh pemanggilan (hapus saat produksi)
if __name__ == "__main__":
    try:
        features_merged = "ini adalah contoh fitur merged"                                  # Ganti dengan fitur merged yang sesuai
        y = "ini adalah contoh target"                                                      # Ganti dengan target yang sesuai

        optimized_models = all_optimization(features_merged, y)                           # lakukan optimasi model
        results_optimized = optimalization_models(optimized_models, features_merged, y)     # lakukan evaluasi model setelah optimasi
        summary_optimalization(results_optimized)                                           # tampilkan rangkuman hasil evaluasi
    except Exception as e:
        print(e)