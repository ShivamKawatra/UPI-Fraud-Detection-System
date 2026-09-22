from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_directories() -> None:
    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def find_dataset_path(dataset_name: str | None = None) -> Path:
    ensure_directories()
    if dataset_name:
        path = RAW_DATA_DIR / dataset_name
        if path.exists():
            return path
        return Path(dataset_name)

    candidates = [
        "upi_transactions_2024.csv",
        "upi_transactions.csv",
        "fraud_upi_data.csv",
        "upi_fraud_data.csv",
        "transaction_data.csv",
    ]
    for name in candidates:
        candidate = RAW_DATA_DIR / name
        if candidate.exists():
            return candidate

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    excel_files = sorted(RAW_DATA_DIR.glob("*.xlsx")) + sorted(RAW_DATA_DIR.glob("*.xls"))
    if excel_files:
        return excel_files[0]

    raise FileNotFoundError("No dataset found in data/raw. Add a CSV/XLS file before running the notebooks.")


def load_dataset(dataset_name: str | None = None) -> Tuple[pd.DataFrame, Path]:
    path = find_dataset_path(dataset_name)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), path
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path), path
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def dataset_report(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "head": df.head().copy(),
        "tail": df.tail().copy(),
        "describe": df.describe(include="all").transpose(),
        "nunique": df.nunique(dropna=True).to_dict(),
    }


def infer_target_column(df: pd.DataFrame) -> str:
    candidate_names = [
        "fraud",
        "is_fraud",
        "fraudulent",
        "fraud_flag",
        "is_fraudulent",
        "label",
        "target",
        "transaction_status",
    ]
    normalized = {str(col).lower(): col for col in df.columns}
    for name in candidate_names:
        if name in normalized:
            return normalized[name]
    for col in df.columns:
        values = set(str(v).strip().lower() for v in df[col].dropna().unique())
        if values and values.issubset({"0", "1", "yes", "no", "true", "false", "fraud", "genuine", "legitimate", "safe", "not_fraud"}):
            return col
    raise ValueError("Could not detect the fraud target column automatically. Inspect the dataset schema and set the target column manually.")


def normalize_target_series(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapping = {
        "fraud": 1,
        "fraudulent": 1,
        "1": 1,
        "yes": 1,
        "true": 1,
        "genuine": 0,
        "legitimate": 0,
        "safe": 0,
        "not_fraud": 0,
        "non_fraud": 0,
        "0": 0,
        "no": 0,
        "false": 0,
        "approved": 0,
        "successful": 0,
        "failed": 1,
    }
    converted = values.map(mapping)
    if converted.isna().any():
        unresolved = sorted(set(values[converted.isna()].unique()))
        raise ValueError(f"Unable to normalize fraud labels: {unresolved}")
    return converted.astype(int)


def clean_dataframe(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    cleaned = df.copy().drop_duplicates().reset_index(drop=True)
    if target_column and target_column in cleaned.columns:
        cleaned[target_column] = normalize_target_series(cleaned[target_column])

    for column in cleaned.columns:
        if "date" in str(column).lower() or "time" in str(column).lower():
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    amount_columns = [
        c for c in cleaned.columns
        if any(token in str(c).lower() for token in ["amount", "value", "amt", "fee", "balance"]) or pd.api.types.is_numeric_dtype(cleaned[c])
    ]
    for column in amount_columns:
        if column == target_column:
            continue
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = cleaned[column].replace([np.inf, -np.inf], np.nan)
        cleaned[column] = cleaned[column].mask(cleaned[column] < 0, np.nan)
        cleaned[column] = cleaned[column].mask(cleaned[column] == 0, np.nan)

    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    numeric_cols = [col for col in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[col])]
    for col in numeric_cols:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    categorical_cols = [col for col in cleaned.columns if col not in numeric_cols and col != target_column]
    for col in categorical_cols:
        cleaned[col] = cleaned[col].fillna("Unknown")

    return cleaned


def remove_obvious_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    leakage_keywords = ["id", "account", "device_id", "session", "customer", "merchant_id", "bank_account", "card"]
    cols_to_drop = []
    for col in df.columns:
        low = str(col).lower()
        if any(keyword in low for keyword in leakage_keywords) and low not in {"device", "bank", "merchant", "network"}:
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop, errors="ignore")


def prepare_features(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    X = df.drop(columns=[target_column], errors="ignore").copy()
    X = remove_obvious_leakage_columns(X)
    y = df[target_column].copy()

    for col in X.columns:
        if "date" in str(col).lower() or "time" in str(col).lower():
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[f"{col}_hour"] = X[col].dt.hour
                X[f"{col}_day"] = X[col].dt.day
                X[f"{col}_dayofweek"] = X[col].dt.dayofweek
                X[f"{col}_month"] = X[col].dt.month
                X[f"{col}_weekend"] = X[col].dt.dayofweek.isin([5, 6]).astype(int)
                X[col] = X[col].dt.strftime("%Y-%m-%d")

    numeric_columns = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_columns = [c for c in X.columns if c not in numeric_columns and X[c].dtype == object]

    for col in numeric_columns:
        if X[col].abs().max() > 1e6:
            X[f"{col}_log"] = np.log1p(X[col].clip(lower=0))

    numeric_columns = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_columns = [c for c in X.columns if c not in numeric_columns and X[c].dtype == object]
    return X, y, numeric_columns, categorical_columns


def build_preprocessor(numeric_columns: List[str], categorical_columns: List[str]) -> ColumnTransformer:
    transformers = []
    if numeric_columns:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No features available for preprocessing.")
    return ColumnTransformer(transformers=transformers)


def build_model_pipeline(model_name: str) -> Pipeline:
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=4000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced", min_samples_leaf=2),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=1.0,
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model name: {model_name}")
    return Pipeline([("preprocessor", None), ("model", models[model_name])])


def split_train_calibration_test(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, train_size=0.5, stratify=y_temp, random_state=42
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
        "PR-AUC": float(average_precision_score(y_true, y_proba)),
        "Specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "FPR": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "FNR": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def train_and_compare_models(
    X_train: pd.DataFrame,
    X_cal: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_cal: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    del X_cal, y_cal
    numeric_columns = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_columns = [c for c in X_train.columns if c not in numeric_columns and X_train[c].dtype == object]

    results: Dict[str, Dict[str, Any]] = {}
    for model_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        pipeline = build_model_pipeline(model_name)
        pipeline.steps[0] = ("preprocessor", build_preprocessor(numeric_columns, categorical_columns))
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = np.asarray(y_pred, dtype=float)

        results[model_name] = {
            "model": pipeline,
            "predictions": y_pred,
            "probabilities": y_proba,
            "true_labels": y_test,
            "metrics": compute_metrics(y_test, y_pred, y_proba),
        }
    return results


def compute_conformal_threshold(model: Any, X_cal: pd.DataFrame, y_cal: pd.Series, alpha: float) -> float:
    probas = model.predict_proba(X_cal)
    class_to_index = {label: idx for idx, label in enumerate(model.classes_)}
    scores = []
    for row, actual in zip(probas, y_cal):
        true_idx = class_to_index[int(actual)]
        scores.append(1.0 - row[true_idx])
    scores = np.asarray(scores)
    return float(np.quantile(scores, 1.0 - alpha, method="linear"))


def generate_prediction_sets(model: Any, X: pd.DataFrame, threshold: float) -> List[set]:
    probas = model.predict_proba(X)
    sets = []
    for row in probas:
        included = {int(label) for label, p in zip(model.classes_, row) if 1.0 - p <= threshold}
        if not included:
            included = {int(model.classes_[np.argmax(row)])}
        sets.append(included)
    return sets


def evaluate_conformal_prediction(prediction_sets: List[set], y_true: pd.Series, target_coverage: float) -> Dict[str, float]:
    coverage = 0
    set_sizes = []
    automatic_predictions = 0
    correct_automatic = 0
    fp_auto = 0
    tn_auto = 0
    fn_auto = 0
    tp_auto = 0

    for pred_set, true_label in zip(prediction_sets, y_true):
        set_sizes.append(len(pred_set))
        if true_label in pred_set:
            coverage += 1
        if len(pred_set) == 1:
            automatic_predictions += 1
            predicted_label = next(iter(pred_set))
            if predicted_label == true_label:
                correct_automatic += 1
            if predicted_label == 1 and true_label == 1:
                tp_auto += 1
            if predicted_label == 1 and true_label == 0:
                fp_auto += 1
            if predicted_label == 0 and true_label == 0:
                tn_auto += 1
            if predicted_label == 0 and true_label == 1:
                fn_auto += 1

    total = len(y_true)
    abstention_rate = float(np.mean([len(s) == 2 for s in prediction_sets]))
    automatic_rate = float(np.mean([len(s) == 1 for s in prediction_sets]))
    selective_accuracy = correct_automatic / automatic_predictions if automatic_predictions else 0.0
    fraud_recall_auto = tp_auto / (tp_auto + fn_auto) if (tp_auto + fn_auto) else 0.0
    auto_fpr = fp_auto / (fp_auto + tn_auto) if (fp_auto + tn_auto) else 0.0
    auto_fnr = fn_auto / (fn_auto + tp_auto) if (fn_auto + tp_auto) else 0.0

    return {
        "target_coverage": float(target_coverage),
        "empirical_coverage": float(coverage / total),
        "coverage_gap": float(target_coverage - (coverage / total)),
        "mean_set_size": float(np.mean(set_sizes)) if set_sizes else 0.0,
        "abstention_rate": abstention_rate,
        "automatic_decision_rate": automatic_rate,
        "manual_review_rate": abstention_rate,
        "selective_accuracy": selective_accuracy,
        "fraud_recall_on_automatic_decisions": fraud_recall_auto,
        "fpr_auto": auto_fpr,
        "fnr_auto": auto_fnr,
    }


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()


def plot_roc_curve(model_results: Dict[str, Dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 6))
    for name, payload in model_results.items():
        y_true = payload["true_labels"]
        y_proba = payload["probabilities"]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()


def plot_precision_recall_curve(model_results: Dict[str, Dict[str, Any]]) -> None:
    plt.figure(figsize=(8, 6))
    for name, payload in model_results.items():
        y_true = payload["true_labels"]
        y_proba = payload["probabilities"]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.plot(recall, precision, label=name)
    plt.title("Precision-Recall Curve Comparison")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()


def plot_metric_bars(model_metrics: Dict[str, Dict[str, float]], metric_names: Iterable[str]) -> None:
    metric_values = {name: {model: metrics[name] for model, metrics in model_metrics.items()} for name in metric_names}
    plt.figure(figsize=(10, 6))
    x = np.arange(len(model_metrics))
    width = 0.8 / len(metric_names)
    for i, metric_name in enumerate(metric_names):
        values = [metric_values[metric_name][model] for model in model_metrics]
        plt.bar(x + i * width, values, width=width, label=metric_name)
    plt.xticks(x + (len(metric_names) - 1) * width / 2, list(model_metrics.keys()))
    plt.title("Model metric comparison")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "FIGURES_DIR",
    "TABLES_DIR",
    "MODELS_DIR",
    "ensure_directories",
    "find_dataset_path",
    "load_dataset",
    "dataset_report",
    "infer_target_column",
    "normalize_target_series",
    "clean_dataframe",
    "remove_obvious_leakage_columns",
    "prepare_features",
    "build_preprocessor",
    "build_model_pipeline",
    "split_train_calibration_test",
    "compute_metrics",
    "train_and_compare_models",
    "compute_conformal_threshold",
    "generate_prediction_sets",
    "evaluate_conformal_prediction",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_metric_bars",
]
