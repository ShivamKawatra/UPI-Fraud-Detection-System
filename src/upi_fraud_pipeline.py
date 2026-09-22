from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

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
    if dataset_name is not None:
        candidate = RAW_DATA_DIR / dataset_name
        if candidate.exists():
            return candidate
        return Path(dataset_name)

    preferred_names = [
        "upi_transactions_2024.csv",
        "upi_transactions.csv",
        "fraud_upi_data.csv",
        "upi_fraud_data.csv",
        "transaction_data.csv",
    ]
    for name in preferred_names:
        candidate = RAW_DATA_DIR / name
        if candidate.exists():
            return candidate

    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    excel_files = sorted(RAW_DATA_DIR.glob("*.xlsx")) + sorted(RAW_DATA_DIR.glob("*.xls"))
    if excel_files:
        return excel_files[0]

    raise FileNotFoundError(
        "No dataset found in data/raw/. Download the public UPI transaction CSV and place it in data/raw/."
    )


def load_dataset(dataset_name: str | None = None) -> Tuple[pd.DataFrame, Path]:
    path = find_dataset_path(dataset_name)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}")
    return df, path


def dataset_diagnostic_report(df: pd.DataFrame) -> Dict[str, Any]:
    report = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "head": df.head().copy(),
        "tail": df.tail().copy(),
        "describe": df.describe(include="all").transpose(),
        "nunique": df.nunique(dropna=True).to_dict(),
    }
    return report


def detect_target_column(df: pd.DataFrame) -> str:
    columns = [col.lower() for col in df.columns]
    target_candidates = [
        "fraud",
        "is_fraud",
        "fraudulent",
        "fraud_flag",
        "label",
        "transaction_status",
        "is_fraudulent",
    ]
    for candidate in target_candidates:
        for col_name in df.columns:
            if str(col_name).lower() == candidate:
                return col_name
        for idx, col_name in enumerate(df.columns):
            lower_name = str(col_name).lower()
            if candidate in lower_name or lower_name.endswith(candidate):
                return col_name

    for col_name in df.columns:
        unique_values = df[col_name].dropna().astype(str).str.lower().unique()
        if set(unique_values) <= {"0", "1", "yes", "no", "true", "false", "fraud", "genuine", "legitimate", "safe", "not_fraud"}:
            return col_name

    raise ValueError(
        "Could not determine the fraud target column automatically. Please inspect the dataset and set the target column explicitly."
    )


def normalize_target_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    mapping = {
        "fraud": 1,
        "fraudulent": 1,
        "1": 1,
        "yes": 1,
        "true": 1,
        "genuine": 0,
        "legitimate": 0,
        "not_fraud": 0,
        "non_fraud": 0,
        "safe": 0,
        "0": 0,
        "no": 0,
        "false": 0,
        "approved": 0,
        "successful": 0,
        "failed": 1,
    }
    normalized = cleaned.map(mapping)
    if normalized.isna().any():
        unresolved = sorted(set(cleaned[normalized.isna()].unique()))
        raise ValueError(f"Target values could not be normalized: {unresolved}")
    return normalized.astype(int)


def identify_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df.columns:
        low = str(col).lower()
        if any(token in low for token in ["amount", "value", "balance", "fee", "score", "count"]):
            numeric_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return list(dict.fromkeys(numeric_cols))


def identify_datetime_columns(df: pd.DataFrame) -> List[str]:
    datetime_candidates = []
    for col in df.columns:
        low = str(col).lower()
        if any(token in low for token in ["date", "time", "timestamp", "datetime", "created_at", "transaction_time"]):
            datetime_candidates.append(col)
    return datetime_candidates


def identify_category_columns(df: pd.DataFrame) -> List[str]:
    excluded = {"id", "transaction_id", "txn_id", "upi_id", "customer_id", "sender_id", "receiver_id"}
    categorical = []
    for col in df.columns:
        low = str(col).lower()
        if low in excluded:
            continue
        if any(token in low for token in ["type", "merchant", "category", "bank", "network", "device", "state", "city", "country", "status"]):
            categorical.append(col)
        elif df[col].dtype == object:
            categorical.append(col)
    return list(dict.fromkeys(categorical))


def remove_obvious_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    leakage_names = [
        "transaction_id",
        "txn_id",
        "upi_id",
        "id",
        "customer_id",
        "sender_id",
        "receiver_id",
        "merchant_id",
        "account_number",
        "bank_account",
        "card_number",
        "device_id",
        "session_id",
    ]
    for col in df.columns:
        low = str(col).lower()
        if low in leakage_names or "_id" in low:
            df = df.drop(columns=[col])
    return df


def clean_dataset(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if target_column is not None and target_column in cleaned.columns:
        cleaned[target_column] = normalize_target_series(cleaned[target_column])

    for col in identify_datetime_columns(cleaned):
        cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")

    amount_candidates = [
        c for c in identify_numeric_columns(cleaned)
        if any(token in str(c).lower() for token in ["amount", "value", "amt", "amount_usd", "transaction_amount"])
    ]
    for col in amount_candidates:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        cleaned[col] = cleaned[col].mask(cleaned[col] < 0, np.nan)
        cleaned[col] = cleaned[col].mask(cleaned[col] == 0, np.nan)

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    numeric_columns = [c for c in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[c])]  
    for col in numeric_columns:
        if cleaned[col].isnull().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    categorical_columns = [c for c in cleaned.columns if c not in numeric_columns and c != target_column]
    for col in categorical_columns:
        cleaned[col] = cleaned[col].fillna("Unknown")

    return cleaned


def make_feature_matrix(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[str]]:
    feature_frame = df.drop(columns=[target_column], errors="ignore").copy()
    feature_frame = remove_obvious_leakage_columns(feature_frame)
    target = df[target_column].copy()

    for col in identify_datetime_columns(feature_frame):
        feature_frame[col] = pd.to_datetime(feature_frame[col], errors="coerce")
        feature_frame[col + "_hour"] = feature_frame[col].dt.hour
        feature_frame[col + "_dayofweek"] = feature_frame[col].dt.dayofweek
        feature_frame[col + "_day"] = feature_frame[col].dt.day
        feature_frame[col + "_month"] = feature_frame[col].dt.month
        feature_frame[col + "_weekend"] = feature_frame[col].dt.dayofweek.isin([5, 6]).astype(int)

    for col in feature_frame.columns:
        if pd.api.types.is_datetime64_any_dtype(feature_frame[col]):
            feature_frame[col] = feature_frame[col].dt.strftime("%Y-%m-%d")

    for col in identify_numeric_columns(feature_frame):
        if feature_frame[col].dtype.kind in {"i", "f"}:
            if feature_frame[col].abs().max() > 1e6:
                feature_frame[col + "_log"] = np.log1p(feature_frame[col].clip(lower=0))

    categorical_columns = [
        c for c in feature_frame.columns
        if c not in identify_numeric_columns(feature_frame)
        and feature_frame[c].dtype == object
    ]
    numeric_columns = [
        c for c in feature_frame.columns
        if c not in categorical_columns and pd.api.types.is_numeric_dtype(feature_frame[c])
    ]

    return feature_frame, target, numeric_columns, categorical_columns, list(feature_frame.columns)


def build_preprocessor(X_train: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("num", numeric_transformer, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_transformer, categorical_columns))

    if not transformers:
        raise ValueError("No feature columns available to preprocess.")

    return ColumnTransformer(transformers=transformers)


def get_model_definitions() -> Dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            scale_pos_weight=1.0,
        ),
    }


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    bacc = balanced_accuracy_score(y_true, y_pred)

    return {
        "Accuracy": float(acc),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "Specificity": float(specificity),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "Balanced Accuracy": float(bacc),
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
    del y_cal
    numeric_columns = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical_columns = [c for c in X_train.columns if c not in numeric_columns and X_train[c].dtype == object]
    results: Dict[str, Dict[str, Any]] = {}

    for model_name, estimator in get_model_definitions().items():
        preprocessor = build_preprocessor(X_train, numeric_columns, categorical_columns)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = np.asarray(y_pred, dtype=float)

        metrics = compute_metrics(y_test, y_pred, y_proba)
        results[model_name] = {
            "model": pipeline,
            "predictions": y_pred,
            "probabilities": y_proba,
            "metrics": metrics,
        }

    return results


def compute_conformal_threshold(model: Any, X_cal: pd.DataFrame, y_cal: pd.Series, alpha: float) -> float:
    probabilities = model.predict_proba(X_cal)
    class_index_map = {label: idx for idx, label in enumerate(model.classes_)}
    scores = []
    for probability_row, true_label in zip(probabilities, y_cal):
        true_idx = class_index_map[int(true_label)]
        scores.append(1.0 - probability_row[true_idx])
    scores = np.asarray(scores, dtype=float)
    return float(np.quantile(scores, 1.0 - alpha, method="linear"))


def generate_prediction_sets(model: Any, X_test: pd.DataFrame, threshold: float) -> List[set]:
    probabilities = model.predict_proba(X_test)
    prediction_sets: List[set] = []
    for probability_row in probabilities:
        included = set()
        for class_label, probability in zip(model.classes_, probability_row):
            if 1.0 - probability <= threshold:
                included.add(int(class_label))
        if not included:
            included.add(int(model.classes_[np.argmax(probability_row)]))
        prediction_sets.append(included)
    return prediction_sets


def evaluate_conformal_prediction(prediction_sets: List[set], y_true: pd.Series, target_coverage: float) -> Dict[str, float]:
    coverage_count = 0
    set_sizes = []
    automatic_decisions = 0
    selective_correct = 0
    total_auto = 0
    tp_auto = 0
    fn_auto = 0
    fp_auto = 0
    tn_auto = 0

    for pred_set, actual in zip(prediction_sets, y_true):
        set_sizes.append(len(pred_set))
        if actual in pred_set:
            coverage_count += 1

        if len(pred_set) == 1:
            automatic_decisions += 1
            predicted_label = next(iter(pred_set))
            if predicted_label == actual:
                selective_correct += 1
            if predicted_label == 1 and actual == 1:
                tp_auto += 1
            if predicted_label == 0 and actual == 1:
                fn_auto += 1
            if predicted_label == 1 and actual == 0:
                fp_auto += 1
            if predicted_label == 0 and actual == 0:
                tn_auto += 1
        
        if len(pred_set) == 2:
            pass

    total_examples = len(y_true)
    empirical_coverage = coverage_count / total_examples
    average_set_size = float(np.mean(set_sizes)) if set_sizes else 0.0
    abstention_rate = float(np.mean([len(s) == 2 for s in prediction_sets]))
    automatic_decision_rate = float(np.mean([len(s) == 1 for s in prediction_sets]))
    selective_accuracy = selective_correct / automatic_decisions if automatic_decisions else 0.0
    manual_review_rate = abstention_rate

    fraud_recall_auto = tp_auto / (tp_auto + fn_auto) if (tp_auto + fn_auto) else 0.0
    auto_fpr = fp_auto / (fp_auto + tn_auto) if (fp_auto + tn_auto) else 0.0
    auto_fnr = fn_auto / (fn_auto + tp_auto) if (fn_auto + tp_auto) else 0.0

    return {
        "target_coverage": float(target_coverage),
        "empirical_coverage": float(empirical_coverage),
        "coverage_gap": float(target_coverage - empirical_coverage),
        "mean_set_size": float(average_set_size),
        "abstention_rate": float(abstention_rate),
        "automatic_decision_rate": float(automatic_decision_rate),
        "selective_accuracy": float(selective_accuracy),
        "manual_review_rate": float(manual_review_rate),
        "fraud_recall_on_automatic_decisions": float(fraud_recall_auto),
        "fpr_auto": float(auto_fpr),
        "fnr_auto": float(auto_fnr),
    }


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()


def plot_roc_curve(results: Dict[str, Dict[str, Any]], figsize=(8, 6)) -> None:
    plt.figure(figsize=figsize)
    for model_name, payload in results.items():
        y_true = payload["y_true"]
        y_proba = payload["probabilities"]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()


def plot_precision_recall_curve(results: Dict[str, Dict[str, Any]], figsize=(8, 6)) -> None:
    plt.figure(figsize=figsize)
    for model_name, payload in results.items():
        y_true = payload["y_true"]
        y_proba = payload["probabilities"]
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.plot(recall, precision, label=model_name)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()


# Imported at module level for plotting functions
from sklearn.metrics import precision_recall_curve, roc_curve
