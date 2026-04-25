# CreditWise Loan Approval System

End-to-end machine learning project that predicts whether a loan application should be approved (`1`) or rejected (`0`) before final human verification.

## Problem Statement

SecureTrust Bank handles high-volume loan applications across urban and rural India. The previous manual screening process was slow and inconsistent, causing:

- false rejections of creditworthy customers (business loss),
- false approvals of risky applicants (financial loss).

This project provides a faster and more consistent pre-screening system built with Python, scikit-learn, FastAPI, and Streamlit.

## Key Results

The tuned model currently reports the following metrics on the sample project dataset:

- Accuracy: `1.00`
- Precision: `1.00`
- Recall: `1.00`
- F1 Score: `1.00`
- ROC-AUC: `1.00`

Best hyperparameters:

- `model__max_depth = 8`
- `model__min_samples_split = 2`
- `model__n_estimators = 200`

> Note: these scores are from a small sample dataset included for demonstration; production performance should be validated on larger real-world data.

## Project Structure

```text
creditwise-loan-approval-ml/
|-- app.py
|-- streamlit_app.py
|-- requirements.txt
|-- README.md
|-- assets/
|-- data/
|   |-- raw/
|   |   `-- loan_applications_sample.csv
|   `-- processed/                  # generated locally, ignored in git
|-- models/                         # generated locally, ignored in git
|-- notebooks/
|   |-- 01_eda.ipynb
|   `-- 02_explainability_bias.ipynb
|-- reports/
|   |-- final_report.md
|   |-- model_comparison.csv
|   |-- best_model_metrics.json
|   `-- fairness_report.json
`-- src/
    |-- __init__.py
    |-- config.py
    |-- preprocess.py
    |-- train_baseline.py
    |-- tune_model.py
    `-- evaluate.py
```

## Dataset Columns

`Applicant_ID`, `Applicant_Income`, `Coapplicant_Income`, `Employment_Status`, `Age`, `Marital_Status`, `Dependents`, `Credit_Score`, `Existing_Loans`, `DTI_Ratio`, `Savings`, `Collateral_Value`, `Loan_Amount`, `Loan_Term`, `Loan_Purpose`, `Property_Area`, `Education_Level`, `Gender`, `Employer_Category`, `Loan_Approved`.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- FastAPI + Uvicorn
- Streamlit
- Jupyter Notebook

## ML Workflow

1. Data loading and validation
2. Preprocessing (imputation, encoding, scaling)
3. Baseline model training
4. Hyperparameter tuning
5. Fairness diagnostics
6. API + dashboard inference

## Run Locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.preprocess
python -m src.train_baseline
python -m src.tune_model
```

Start backend API:

```bash
uvicorn app:app --reload
```

Start frontend dashboard:

```bash
streamlit run streamlit_app.py
```

## API Reference

- `GET /health` - health check and model availability
- `POST /predict` - returns:
  - `prediction` (`0` or `1`)
  - `decision` (`Approved` or `Rejected`)
  - `approval_probability`

## Dashboard Features

- Minimal loan application form UI
- Decision and probability output
- Risk accelerator visualization (High -> Medium -> Low)
- Eligibility flags and actionable suggestions
- Factor influence and score charts

## Fairness and Explainability

`notebooks/02_explainability_bias.ipynb` and `reports/fairness_report.json` include:

- grouped approval-rate checks,
- grouped TPR/FPR checks,
- feature influence review.

## Screenshots

Add your UI screenshots in `assets/` and link them here:

- `assets/dashboard-form.png`
- `assets/dashboard-result.png`
- `assets/risk-accelerator.png`

## Roadmap

- Integrate production dataset and validation pipeline
- Add SHAP-based local explanations
- Add CI/CD and container deployment
- Add model monitoring and drift alerts
