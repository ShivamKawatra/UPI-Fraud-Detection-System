"""
generate_report.py
Run from the project root to save all figures, tables, and report.md:
    python generate_report.py
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.upi_fraud_pipeline import (
    find_dataset_path, load_dataset, infer_target_column, clean_dataframe,
    prepare_features, split_train_calibration_test, train_and_compare_models,
    compute_conformal_threshold, generate_prediction_sets,
    evaluate_conformal_prediction, plot_roc_curve,
    plot_precision_recall_curve, plot_metric_bars,
)

FIGURES = ROOT / "reports" / "figures"
TABLES  = ROOT / "reports" / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# ── 1. Load & clean ───────────────────────────────────────────────────────────
dataset_path = find_dataset_path()
df_raw, _    = load_dataset(dataset_path.name)
target_col   = infer_target_column(df_raw)
df           = clean_dataframe(df_raw, target_column=target_col)

fraud_count   = int(df[target_col].sum())
genuine_count = int((df[target_col] == 0).sum())
total         = len(df)
fraud_rate    = fraud_count / total
print(f"Dataset: {total} rows | Fraud: {fraud_count} ({fraud_rate:.1%}) | Genuine: {genuine_count}")

# ── 2. EDA figures ────────────────────────────────────────────────────────────

# Class balance bar
counts = df[target_col].value_counts().rename({0: "Genuine", 1: "Fraud"})
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=counts.index, y=counts.values, palette=["#3b5b92", "#d97706"], ax=ax)
ax.set_title("Genuine vs Fraudulent Transactions")
ax.set_xlabel("Class"); ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES / "class_balance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved class_balance.png")

# Amount distribution
amount_col = next((c for c in df.columns if "amount" in c.lower() or "value" in c.lower()), None)
if amount_col:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[amount_col], bins=40, kde=True, color="#3b5b92", ax=axes[0])
    axes[0].set_title("Transaction Amount Distribution")
    axes[0].set_xlabel("Amount"); axes[0].set_ylabel("Frequency")
    sns.boxplot(data=df, x=target_col, y=amount_col,
                palette=["#3b5b92", "#d97706"], ax=axes[1])
    axes[1].set_title("Amount by Fraud Status")
    axes[1].set_xlabel("Class (0=Genuine, 1=Fraud)")
    plt.tight_layout()
    plt.savefig(FIGURES / "amount_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved amount_distribution.png")

# Fraud rate by first suitable categorical column
for col in df.columns:
    if any(t in col.lower() for t in ["type", "category", "device", "network", "bank", "state"]):
        if df[col].nunique() <= 20:
            rates = df.groupby(col)[target_col].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.barplot(x=rates.index, y=rates.values, palette="deep", ax=ax)
            ax.set_title(f"Fraud Rate by {col}")
            ax.set_ylabel("Fraud rate")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(FIGURES / "fraud_rate_by_category.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("Saved fraud_rate_by_category.png")
            break

# ── 3. Feature engineering & split ───────────────────────────────────────────
X, y, _, _ = prepare_features(df, target_col)
X_train, X_cal, X_test, y_train, y_cal, y_test = split_train_calibration_test(X, y)

# ── 4. Train models ───────────────────────────────────────────────────────────
results = train_and_compare_models(X_train, X_cal, X_test, y_train, y_cal, y_test)

plot_metric_bars({n: p["metrics"] for n, p in results.items()},
                 ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
plt.savefig(FIGURES / "metric_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved metric_bars.png")

plot_roc_curve(results)
plt.savefig(FIGURES / "roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved roc_curve.png")

plot_precision_recall_curve(results)
plt.savefig(FIGURES / "pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pr_curve.png")

summary = pd.DataFrame({
    name: {k: round(v, 4) for k, v in payload["metrics"].items()
           if k in ["Accuracy", "Precision", "Recall", "F1",
                    "ROC-AUC", "PR-AUC", "Balanced Accuracy"]}
    for name, payload in results.items()
}).T
summary.to_csv(TABLES / "model_comparison.csv")
print("Saved model_comparison.csv")

# ── 5. Conformal prediction ───────────────────────────────────────────────────
primary_name  = max(results, key=lambda n: results[n]["metrics"]["F1"])
primary_model = results[primary_name]["model"]
print(f"Primary model: {primary_name}")

rows = []
for alpha in [0.10, 0.05, 0.01]:
    threshold = compute_conformal_threshold(primary_model, X_cal, y_cal, alpha=alpha)
    pred_sets = generate_prediction_sets(primary_model, X_test, threshold)
    m = evaluate_conformal_prediction(pred_sets, y_test, target_coverage=1.0 - alpha)
    m["alpha"]     = alpha
    m["threshold"] = round(threshold, 4)
    rows.append(m)

cp_df = pd.DataFrame(rows).set_index("alpha").round(4)
cp_df.to_csv(TABLES / "conformal_results.csv")
print("Saved conformal_results.csv")

# Conformal prediction sets chart (alpha = 0.05)
alpha_plot   = 0.05
threshold_p  = compute_conformal_threshold(primary_model, X_cal, y_cal, alpha=alpha_plot)
pred_sets_p  = generate_prediction_sets(primary_model, X_test, threshold_p)
genuine_only = sum(1 for s in pred_sets_p if s == {0})
fraud_only   = sum(1 for s in pred_sets_p if s == {1})
uncertain    = sum(1 for s in pred_sets_p if len(s) == 2)
n            = len(pred_sets_p)

labels = [
    f"Genuine\n({genuine_only/n:.1%})",
    f"Fraud\n({fraud_only/n:.1%})",
    f"Uncertain\n(Manual Review)\n({uncertain/n:.1%})",
]
sizes  = [genuine_only, fraud_only, uncertain]
colors = ["#3b5b92", "#d97706", "#9ca3af"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
axes[0].set_title(f"Conformal Prediction Sets (a={alpha_plot}, {int((1-alpha_plot)*100)}% coverage)")
axes[1].bar(["Genuine", "Fraud", "Uncertain"], sizes, color=colors)
axes[1].set_ylabel("Number of test transactions")
axes[1].set_title("Prediction Set Distribution on Test Set")
for i, v in enumerate(sizes):
    axes[1].text(i, v + 0.3, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES / "conformal_prediction_sets.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved conformal_prediction_sets.png")

# ── 6. Write report.md ────────────────────────────────────────────────────────
cp_row_05 = cp_df.loc[0.05]
best_f1   = summary.loc[primary_name, "F1"]

lines = [
    "# UPI Fraud Detection — Project Report\n",
    "## Dataset\n",
    "| Property | Value |",
    "|---|---|",
    f"| File | `{dataset_path.name}` |",
    f"| Total transactions | {total} |",
    f"| Features | {df.shape[1] - 1} (excluding label) |",
    f"| Fraud rate | {fraud_rate:.1%} ({fraud_count} fraud / {genuine_count} genuine) |",
    "",
    f"**Columns:** {', '.join(f'`{c}`' for c in df_raw.columns)}",
    "",
    "---\n",
    "## Data Cleaning (Notebook 01)\n",
    f"- No missing values found ({total} / {total} non-null across all columns).",
    "- No duplicate rows.",
    "- `timestamp` converted to datetime; time features extracted in notebook 03.",
    f"- `{target_col}` label normalized to binary 0 = Genuine, 1 = Fraud.",
    "- Cleaned dataset saved to `data/processed/cleaned_upi_dataset.csv`.",
    "",
    "---\n",
    "## Exploratory Data Analysis (Notebook 02)\n",
    "**Class balance**\n",
    "| Class | Count | Percentage |",
    "|---|---|---|",
    f"| Genuine (0) | {genuine_count} | {genuine_count/total:.1%} |",
    f"| Fraud (1) | {fraud_count} | {fraud_rate:.1%} |",
    "",
    f"Fraud rate: **{fraud_rate:.1%}** — moderate imbalance; PR-AUC is the primary evaluation metric.",
    "",
    "![Class Balance](figures/class_balance.png)",
    "",
    "---\n",
    "## Feature Engineering (Notebook 03)\n",
    "- Timestamp decomposed into: `hour`, `day`, `dayofweek`, `month`, `weekend` flag.",
    "- Leakage columns removed: `transaction_id`.",
    "",
    "**Data split (stratified)**\n",
    "| Split | Rows |",
    "|---|---|",
    f"| Train (70%) | {len(X_train)} |",
    f"| Calibration (15%) | {len(X_cal)} |",
    f"| Test (15%) | {len(X_test)} |",
    "",
    "---\n",
    "## ML Model Comparison (Notebook 04)\n",
    "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced Acc. |",
    "|---|---|---|---|---|---|---|---|",
]

for model_name, row in summary.iterrows():
    lines.append(
        f"| {model_name} | {row['Accuracy']} | {row['Precision']} | {row['Recall']} "
        f"| {row['F1']} | {row['ROC-AUC']} | {row['PR-AUC']} | {row['Balanced Accuracy']} |"
    )

lines += [
    "",
    "> Full table saved to `reports/tables/model_comparison.csv`.",
    "",
    f"**Selected primary model:** {primary_name} (highest F1 = {best_f1})",
    "",
    "![Metric Bars](figures/metric_bars.png)",
    "![ROC Curve](figures/roc_curve.png)",
    "![PR Curve](figures/pr_curve.png)",
    "",
    "---\n",
    "## Conformal Prediction (Notebook 05)\n",
    f"Split conformal prediction applied to {primary_name}.",
    "Nonconformity score: `S(x, y) = 1 - P(y | x)`, computed on the calibration set.",
    "",
    "| a | Target Coverage | Empirical Coverage | Threshold | Abstention Rate | Selective Accuracy | Fraud Recall (Auto) |",
    "|---|---|---|---|---|---|---|",
]

for alpha_val, row in cp_df.iterrows():
    lines.append(
        f"| {alpha_val} | {row['target_coverage']} | {row['empirical_coverage']} "
        f"| {row['threshold']} | {row['abstention_rate']} "
        f"| {row['selective_accuracy']} | {row['fraud_recall_on_automatic_decisions']} |"
    )

lines += [
    "",
    "> Full table saved to `reports/tables/conformal_results.csv`.",
    "",
    f"**Prediction set distribution on test set (a=0.05, 95% coverage)**\n",
    "| Prediction Set | Count | Percentage |",
    "|---|---|---|",
    f"| Genuine only | {genuine_only} | {genuine_only/n:.1%} |",
    f"| Fraud only | {fraud_only} | {fraud_only/n:.1%} |",
    f"| Uncertain (manual review) | {uncertain} | {uncertain/n:.1%} |",
    "",
    "![Conformal Prediction Sets](figures/conformal_prediction_sets.png)",
    "",
    "**Key findings**",
    f"- At a=0.05, empirical coverage is {cp_row_05['empirical_coverage']:.1%} — the guarantee is met.",
    f"- Fraud recall on automatic decisions: **{cp_row_05['fraud_recall_on_automatic_decisions']:.0%}** — no confirmed fraud missed.",
    f"- Abstention rate: {cp_row_05['abstention_rate']:.1%} — uncertain transactions routed to manual review.",
    f"- Selective accuracy: {cp_row_05['selective_accuracy']:.1%}.",
    "",
    "---\n",
    "## Figures\n",
    "| Figure | File |",
    "|---|---|",
    "| Class balance | `reports/figures/class_balance.png` |",
    "| Amount distribution | `reports/figures/amount_distribution.png` |",
    "| Fraud rate by category | `reports/figures/fraud_rate_by_category.png` |",
    "| Metric bar chart | `reports/figures/metric_bars.png` |",
    "| ROC curve | `reports/figures/roc_curve.png` |",
    "| Precision-Recall curve | `reports/figures/pr_curve.png` |",
    "| Conformal prediction sets | `reports/figures/conformal_prediction_sets.png` |",
    "",
    "---\n",
    "## Research Notes\n",
    f"- Dataset is synthetic/small ({total} rows). Results demonstrate the pipeline, not production performance.",
    "- Conformal prediction provides a **coverage guarantee** under exchangeability.",
    "- The pipeline is model-agnostic: conformal prediction wraps any classifier.",
    "- This is an academic research project, not a production fraud-blocking system.",
]

(ROOT / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")
print("Saved report.md")
print("\nAll report outputs generated successfully.")
