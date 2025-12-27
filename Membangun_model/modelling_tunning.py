import pandas as pd
import os
import mlflow
import mlflow.sklearn
import tempfile
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# Konfigurasi Tampilan dan MLflow
pd.set_option('display.max_columns', None)

# Pastikan tidak ada konflik environment
if "MLFLOW_EXPERIMENT_ID" in os.environ:
    del os.environ["MLFLOW_EXPERIMENT_ID"]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Random Forest Diabetes Classification")

# Load Data
df = pd.read_csv("/Users/Shared/Files From d.localized/Rosita/DICODING ASAH 2025/Proyek Membangun Sistem Machine Learning/Membangun_model/diabetes_dataset_2019_preprocessing.csv")

# Pisahkan fitur dan target
X = df.drop(columns=['Diabetic', 'is_train'])
y = df['Diabetic']

# Split menjadi train dan test berdasarkan kolom 'is_train'
train_mask = df['is_train'] == 1
test_mask = df['is_train'] == 0

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print("Training class distribution:\n", y_train.value_counts())
print("\nTest class distribution:\n", y_test.value_counts())

# Encode Target (LabelEncoder untuk 'yes'/'no')
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Setup Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced']
}

# Gunakan StratifiedKFold untuk menjaga distribusi kelas
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Eksperimen Manual Logging
with mlflow.start_run(run_name="RandomForest_Diabetes_ManualLog") as run:
    # Latih model dengan GridSearchCV
    grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring='f1_macro',
        cv=skf,
        n_jobs=-1
    )
    grid.fit(X_train, y_train_enc)
    best_model = grid.best_estimator_

    # Prediksi pada data test
    y_pred_enc = best_model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    # Hitung Metrik
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test_enc, best_model.predict_proba(X_test)[:, 1])


    # Metrik tambahan: Precision-Recall AUC
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test_enc, best_model.predict_proba(X_test)[:, 1]
    )
    pr_auc = auc(recall_curve, precision_curve)

    # Log Parameter dan Metrik
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1_macro", f1_macro)
    mlflow.log_metric("test_precision_macro", precision_macro)
    mlflow.log_metric("test_recall_macro", recall_macro)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("pr_auc", pr_auc)

    # Log F1 per class
    f1_per_class = f1_score(y_test, y_pred, average=None)
    for idx, class_name in enumerate(le.classes_):
        mlflow.log_metric(f"f1_{class_name.lower()}", f1_per_class[idx])

    # Simpan Artefak
    # 1. HTML: Classification Report
    report_text = classification_report(y_test, y_pred)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Estimator Report - Diabetes Prediction</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h2 {{ color: #2c3e50; }}
            pre {{ background: #f8f9fa; padding: 10px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h2>Classification Report</h2>
        <pre>{report_text}</pre>
    </body>
    </html>
    """
    with tempfile.TemporaryDirectory() as tmp:
        html_path = os.path.join(tmp, "estimator.html")
        with open(html_path, "w") as f:
            f.write(html_content)
        mlflow.log_artifact(html_path)

    # 2. JSON: Informasi metrik dan eksperimen
    metric_info = {
        "metrics": {
            "test_accuracy": acc,
            "test_f1_macro": f1_macro,
            "test_precision_macro": precision_macro,
            "test_recall_macro": recall_macro,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        },
        "f1_per_class": {
            class_name.lower(): float(f1_per_class[i])
            for i, class_name in enumerate(le.classes_)
        },
        "best_parameters": grid.best_params_,
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "target_classes": le.classes_.tolist()
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "metric_info.json")
        with open(json_path, "w") as f:
            json.dump(metric_info, f, indent=2)
        mlflow.log_artifact(json_path)

    # 3. Confusion Matrix (PNG)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    with tempfile.TemporaryDirectory() as tmp:
        cm_path = os.path.join(tmp, "training_confusion_matrix.png")
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
    plt.close()

    # Simpan Model
    mlflow.sklearn.log_model(best_model, "model")

    # === Informasi Tambahan ===
    print(f"\nRun ID: {run.info.run_id}")
    print("Semua metrik, artefak, dan model telah disimpan ke MLflow.")