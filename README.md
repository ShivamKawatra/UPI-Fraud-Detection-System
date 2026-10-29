# Uncertainty-Aware UPI Fraud Detection Using Machine Learning and Conformal Prediction

This project is a research-oriented, Python-only machine learning workflow for UPI transaction fraud detection. It combines:

- data cleaning and auditing
- exploratory data analysis
- visualization with Matplotlib and Seaborn
- feature engineering and preprocessing
- standard binary classification models (Logistic Regression, Random Forest, XGBoost)
- conformal prediction for uncertainty-aware classification

The project is intentionally kept simple and focused on notebooks and Python scripts. It does not include dashboards, APIs, or deployment components.

## Project structure

```text
upi-fraud-detection/
├── data/
│   ├── raw/
│   │   └── upi_transactions_2024.csv
│   └── processed/
│       └── cleaned_upi_dataset.csv
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_conformal_prediction.ipynb
├── reports/
│   ├── figures/
│   │   ├── class_balance.png
│   │   ├── amount_distribution.png
│   │   ├── fraud_rate_by_category.png
│   │   ├── metric_bars.png
│   │   ├── roc_curve.png
│   │   ├── pr_curve.png
│   │   └── conformal_prediction_sets.png
│   ├── tables/
│   │   ├── model_comparison.csv
│   │   └── conformal_results.csv
│   └── report.md
├── models/
├── src/
│   ├── __init__.py
│   └── upi_fraud_pipeline.py
├── generate_report.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Dataset requirement

Use a public UPI transaction dataset such as the Kaggle dataset referenced in the project specification. Place the CSV file in the `data/raw/` directory before running the notebooks.

Accepted dataset examples:

- `upi_transactions_2024.csv`
- `upi_transactions.csv`
- `fraud_upi_data.csv`

The code automatically detects the fraud label column and main transaction fields from the dataset schema.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run the notebooks

```bash
jupyter notebook notebooks/
```

Run notebooks in order:

1. `01_data_cleaning.ipynb` — loads raw data, removes duplicates, normalizes labels, saves cleaned CSV
2. `02_eda_visualization.ipynb` — class balance, amount distributions, fraud rate by category
3. `03_feature_engineering.ipynb` — timestamp decomposition, leakage removal, train/calibration/test split
4. `04_ml_models.ipynb` — trains Logistic Regression, Random Forest, XGBoost; compares metrics
5. `05_conformal_prediction.ipynb` — wraps best model with split conformal prediction; outputs genuine/fraud/uncertain sets

## Generate report and save all figures

To save all figures, tables, and a markdown report without re-running notebooks:

```bash
python generate_report.py
```

This script runs the full pipeline programmatically and writes the following outputs:

**Figures** (`reports/figures/`)

| File | Description |
|---|---|
| `class_balance.png` | Genuine vs fraud transaction counts |
| `amount_distribution.png` | Transaction amount histogram and boxplot by class |
| `fraud_rate_by_category.png` | Fraud rate broken down by a categorical feature |
| `metric_bars.png` | Precision, Recall, F1, ROC-AUC, PR-AUC per model |
| `roc_curve.png` | ROC curves for all three models |
| `pr_curve.png` | Precision-Recall curves for all three models |
| `conformal_prediction_sets.png` | Genuine / Fraud / Uncertain split at α=0.05 |

**Tables** (`reports/tables/`)

| File | Description |
|---|---|
| `model_comparison.csv` | Full metrics table for all three models |
| `conformal_results.csv` | Coverage, abstention rate, selective accuracy at α=0.10, 0.05, 0.01 |

**Report** — `reports/report.md` with all real data values embedded.

## Source module

`src/upi_fraud_pipeline.py` is the shared backend imported by all notebooks and `generate_report.py`. It contains all reusable functions for data loading, cleaning, feature engineering, model training, conformal prediction, and plotting. Notebooks do not duplicate logic — they call functions from this module.

## Key results (on current dataset)

- Dataset: 300 rows, 12% fraud rate (36 fraud / 264 genuine)
- Best model: Logistic Regression (F1 = 0.615)
- Conformal prediction at α=0.05: 95.6% empirical coverage, 31.1% abstention rate, 100% fraud recall on automatic decisions

## Important research notes

- The project uses public or synthetic datasets only.
- It does not claim access to real NPCI, bank, or live UPI transaction data.
- Conformal prediction is implemented as an uncertainty-aware post-processing layer on top of a trained classifier.
- This is a research and academic project, not a production fraud-blocking system.
- The empirical coverage depends on data quality, exchangeability assumptions, and possible distribution shift.

## License

This project is intended for academic and research use.
