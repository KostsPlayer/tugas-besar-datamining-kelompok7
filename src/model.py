# src/model.py

"""
model.py

Module ini digunakan untuk fungsi dari pemodelan.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, classification_report
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from yellowbrick.classifier import ROCAUC

import mlflow
from mlflow.models.signature import infer_signature

# Dictionary model
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "LogReg": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "MLP": MLPClassifier(max_iter=1000, random_state=42)
}

def directional_accuracy(y_true, y_pred):
    """
    Mengukur kesesuaian arah perubahan antara y_true dan y_pred.
    """
    return np.mean(np.sign(y_true.diff().fillna(0)) == np.sign(pd.Series(y_pred).diff().fillna(0)))

def evaluate_model(model_or_pipeline, X, y, model_name="Model"):
    """
    Evaluasi model menggunakan TimeSeriesSplit dengan berbagai metrik dan visualisasi.

    Parameters:
        model_or_pipeline: Estimator atau pipeline sklearn
        X (pd.DataFrame): Fitur
        y (pd.Series): Target
        model_name (str): Nama model untuk pelaporan

    Returns:
        dict: Rata-rata dan standar deviasi dari metrik evaluasi
    """
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'directional_acc': []}
    all_conf_matrices = []

    fold = 1
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Buat pipeline baru jika bukan pipeline
        if isinstance(model_or_pipeline, Pipeline):
            pipeline = model_or_pipeline
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model_or_pipeline)
            ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if hasattr(pipeline.named_steps['model'], "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline.named_steps['model'], "decision_function"):
            y_proba = pipeline.decision_function(X_test)
        else:
            y_proba = None

        if len(np.unique(y_test)) > 1:
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            da = directional_accuracy(y_test.reset_index(drop=True), y_pred)

            roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0

            metrics['accuracy'].append(acc)
            metrics['precision'].append(prec)
            metrics['recall'].append(rec)
            metrics['f1'].append(f1)
            metrics['roc_auc'].append(roc_auc)
            metrics['directional_acc'].append(da)
            all_conf_matrices.append(cm)

            print(f"\n📊 Fold {fold} Confusion Matrix:")
            print(classification_report(y_test, y_pred, zero_division=0))

            # Confusion Matrix plot
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{model_name} - Fold {fold} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.show()

            # ROC Curve via Yellowbrick
            if y_proba is not None:
                visualizer = ROCAUC(pipeline.named_steps['model'], classes=["Neg", "Pos"], binary=True)
                visualizer.fit(X_train, y_train)
                visualizer.score(X_test, y_test)
                visualizer.show()
        else:
            print(f"\n⚠️ Fold {fold}: Only one class present in y_test, skipping metrics.")
        
        fold += 1

    # Hitung rata-rata dan standar deviasi metrik
    mean_metrics = {f"{k}_mean": np.mean(v) if v else 0 for k, v in metrics.items()}
    std_metrics = {f"{k}_std": np.std(v) if v else 0 for k, v in metrics.items()}

    # MLflow logging
    with mlflow.start_run(run_name=model_name):
        for k, v in mean_metrics.items():
            mlflow.log_metric(k, v)
        for k, v in std_metrics.items():
            mlflow.log_metric(k, v)

        input_example = X_test.head(1)
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(pipeline, f"{model_name}_model", input_example=input_example, signature=signature)

    return {**mean_metrics, **std_metrics}


# Contoh pemanggilan (hapus saat produksi)
if __name__ == "__main__":
    try:
        X = "ini adalah contoh fitur-fitur model"  # Ganti dengan fitur-fitur yang sesuai
        y = "ini adalah contoh target"      # Ganti dengan target yang sesuai

        results = evaluate_model(models['RandomForest'], X, y, model_name="RandomForest")
        print(results)
    except Exception as e:
        print("Terjadi kesalahan saat evaluasi:", e)
