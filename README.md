# Uncertainty-Aware UPI Fraud Detection Using Machine Learning and Conformal Prediction

This project is a research-oriented, Python-only machine learning workflow for UPI transaction fraud detection. It combines:

- data cleaning and auditing
- exploratory data analysis
- visualization with Matplotlib and Seaborn
- feature engineering and preprocessing
- standard binary classification models
- conformal prediction for uncertainty-aware classification

The project is intentionally kept simple and focused on notebooks and Python scripts. It does not include dashboards, APIs, or deployment components.

## Project structure

```text
upi-fraud-detection/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_conformal_prediction.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── models/
├── src/
│   ├── __init__.py
│   └── upi_fraud_pipeline.py
├── .gitignore
├── requirements.txt
├── README.md
└── .
```

## Dataset requirement

Use a public UPI transaction dataset such as the Kaggle dataset referenced in the project specification. Place the CSV file in the `data/raw/` directory before running the notebooks.

Accepted dataset examples:

- `upi_transactions_2024.csv`
- `upi_transactions.csv`
- `fraud_upi_data.csv`

The code is written to automatically detect the fraud label column and the main transaction fields from the actual dataset schema.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run the notebooks

```bash
jupyter notebook notebooks/
```

Then run notebooks in order:

1. `01_data_cleaning.ipynb`
2. `02_eda_visualization.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_ml_models.ipynb`
5. `05_conformal_prediction.ipynb`

## Important research notes

- The project uses public or synthetic datasets only.
- It does not claim access to real NPCI, bank, or live UPI transaction data.
- Conformal prediction is implemented as an uncertainty-aware post-processing layer on top of a trained classifier.
- This is a research and academic project, not a production fraud-blocking system.
- The empirical coverage depends on data quality, exchangeability assumptions, and possible distribution shift.

## License

This project is intended for academic and research use.
