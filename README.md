# CreditWise Loan Approval System

Machine learning project for loan pre-screening before final human verification.

Given applicant financial and profile details, the model predicts whether a loan should be approved (`1`) or rejected (`0`) and provides risk-focused insights on the dashboard.

## Problem Statement

In the given business scenario (SecureTrust Bank), manual loan verification leads to two common issues:

- some good applicants get rejected,
- some high-risk applicants get approved.

The goal of this project is to make that first-level screening faster and more consistent using machine learning.

## Key Results

On the current sample dataset in this repository, the tuned model gives:

- Accuracy: `1.00`
- Precision: `1.00`
- Recall: `1.00`
- F1 Score: `1.00`
- ROC-AUC: `1.00`

Best hyperparameters:

- `model__max_depth = 8`
- `model__min_samples_split = 2`
- `model__n_estimators = 200`

Important note: these scores are from a small demo dataset, so they should not be treated as production-grade performance.

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

## Tech Stack Used

- Python
- pandas, numpy
- scikit-learn
- FastAPI + Uvicorn
- Streamlit
- Jupyter Notebook

## What This Repo Contains

- `src/preprocess.py` - data split and preprocessing setup
- `src/train_baseline.py` - baseline models + comparison report
- `src/tune_model.py` - hyperparameter tuning + best model export
- `app.py` - FastAPI inference endpoints
- `streamlit_app.py` - frontend dashboard for interactive prediction
- `notebooks/` - EDA and fairness/explainability notebooks
- `reports/` - saved model metrics and fairness outputs

## Run Locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m src.preprocess
python -m src.train_baseline
python -m src.tune_model
```

Run API:

```bash
uvicorn app:app --reload
```

Run dashboard:

```bash
streamlit run streamlit_app.py
```

## API Reference

- `GET /health` - health check and model availability
- `POST /predict` - returns `prediction`, `decision`, and `approval_probability`

## Dashboard Features

- clean form-based input for applicant details
- prediction summary (approval/rejection + probability)
- risk accelerator bar (High -> Medium -> Low)
- rule-based eligibility flags
- actionable suggestions based on risk signals
- factor influence table + score chart

## Fairness and Explainability

`notebooks/02_explainability_bias.ipynb` and `reports/fairness_report.json` include:

- group-level approval checks
- group-level TPR/FPR checks
- model behavior summary by selected segments

## Screenshots

You can add UI screenshots in `assets/` and link them below:

- `assets/dashboard-form.png`
- `assets/dashboard-result.png`
- `assets/risk-accelerator.png`

## Limitations

- small demo dataset (not a real production banking dataset)
- no authentication layer on API/dashboard
- no CI pipeline yet

## Next Improvements

- train and validate on larger real-world data
- add SHAP for per-applicant explanation
- add CI/CD + deployment
- add drift/performance monitoring
