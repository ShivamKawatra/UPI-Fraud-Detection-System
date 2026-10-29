# UPI Fraud Detection — Project Report

## Dataset

| Property | Value |
|---|---|
| File | `upi_transactions_2024.csv` |
| Total transactions | 300 |
| Features | 8 (excluding label) |
| Fraud rate | 12.0% (36 fraud / 264 genuine) |

**Columns:** `transaction_id`, `timestamp`, `amount`, `merchant_category`, `transaction_type`, `device_type`, `network_type`, `sender_state`, `fraud`

---

## Data Cleaning (Notebook 01)

- No missing values found (300 / 300 non-null across all columns).
- No duplicate rows.
- `timestamp` converted to datetime; time features extracted in notebook 03.
- `fraud` label normalized to binary 0 = Genuine, 1 = Fraud.
- Cleaned dataset saved to `data/processed/cleaned_upi_dataset.csv`.

---

## Exploratory Data Analysis (Notebook 02)

**Class balance**

| Class | Count | Percentage |
|---|---|---|
| Genuine (0) | 264 | 88.0% |
| Fraud (1) | 36 | 12.0% |

Fraud rate: **12.0%** — moderate imbalance; PR-AUC is the primary evaluation metric.

![Class Balance](figures/class_balance.png)

---

## Feature Engineering (Notebook 03)

- Timestamp decomposed into: `hour`, `day`, `dayofweek`, `month`, `weekend` flag.
- Leakage columns removed: `transaction_id`.

**Data split (stratified)**

| Split | Rows |
|---|---|
| Train (70%) | 210 |
| Calibration (15%) | 45 |
| Test (15%) | 45 |

---

## ML Model Comparison (Notebook 04)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced Acc. |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8889 | 0.5 | 0.8 | 0.6154 | 0.925 | 0.6769 | 0.85 |
| Random Forest | 0.9333 | 1.0 | 0.4 | 0.5714 | 0.935 | 0.8556 | 0.7 |
| XGBoost | 0.8889 | 0.0 | 0.0 | 0.0 | 0.945 | 0.8048 | 0.5 |

> Full table saved to `reports/tables/model_comparison.csv`.

**Selected primary model:** Logistic Regression (highest F1 = 0.6154)

![Metric Bars](figures/metric_bars.png)
![ROC Curve](figures/roc_curve.png)
![PR Curve](figures/pr_curve.png)

---

## Conformal Prediction (Notebook 05)

Split conformal prediction applied to Logistic Regression.
Nonconformity score: `S(x, y) = 1 - P(y | x)`, computed on the calibration set.

| a | Target Coverage | Empirical Coverage | Threshold | Abstention Rate | Selective Accuracy | Fraud Recall (Auto) |
|---|---|---|---|---|---|---|
| 0.1 | 0.9 | 0.9556 | 0.7607 | 0.2667 | 0.9394 | 1.0 |
| 0.05 | 0.95 | 0.9556 | 0.8117 | 0.3111 | 0.9355 | 1.0 |
| 0.01 | 0.99 | 0.9778 | 0.8624 | 0.3333 | 0.9667 | 1.0 |

> Full table saved to `reports/tables/conformal_results.csv`.

**Prediction set distribution on test set (a=0.05, 95% coverage)**

| Prediction Set | Count | Percentage |
|---|---|---|
| Genuine only | 27 | 60.0% |
| Fraud only | 4 | 8.9% |
| Uncertain (manual review) | 14 | 31.1% |

![Conformal Prediction Sets](figures/conformal_prediction_sets.png)

**Key findings**
- At a=0.05, empirical coverage is 95.6% — the guarantee is met.
- Fraud recall on automatic decisions: **100%** — no confirmed fraud missed.
- Abstention rate: 31.1% — uncertain transactions routed to manual review.
- Selective accuracy: 93.5%.

---

## Figures

| Figure | File |
|---|---|
| Class balance | `reports/figures/class_balance.png` |
| Amount distribution | `reports/figures/amount_distribution.png` |
| Fraud rate by category | `reports/figures/fraud_rate_by_category.png` |
| Metric bar chart | `reports/figures/metric_bars.png` |
| ROC curve | `reports/figures/roc_curve.png` |
| Precision-Recall curve | `reports/figures/pr_curve.png` |
| Conformal prediction sets | `reports/figures/conformal_prediction_sets.png` |

---

## Research Notes

- Dataset is synthetic/small (300 rows). Results demonstrate the pipeline, not production performance.
- Conformal prediction provides a **coverage guarantee** under exchangeability.
- The pipeline is model-agnostic: conformal prediction wraps any classifier.
- This is an academic research project, not a production fraud-blocking system.