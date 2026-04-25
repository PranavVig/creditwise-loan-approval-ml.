# CreditWise Loan Approval System

Machine Learning based loan approval prediction system for personal and home loans.

## Business Context

SecureTrust Bank receives a high volume of daily loan applications from urban and rural regions. The existing manual process is slow and inconsistent, which creates two major problems:

- Good customers can be rejected (lost business).
- High-risk customers can be approved (financial loss).

This project builds an intelligent loan approval system that predicts whether an application should be approved (`1`) or rejected (`0`) before final human verification.

## Project Structure

```text
creditwise-loan-approval-ml/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   |-- raw/
|   |   `-- loan_applications_sample.csv
|   `-- processed/
|-- models/
|-- notebooks/
|   |-- 01_eda.ipynb
|   `-- 02_explainability_bias.ipynb
|-- reports/
|   `-- final_report.md
`-- src/
    |-- __init__.py
    |-- config.py
    |-- preprocess.py
    |-- train_baseline.py
    |-- tune_model.py
    `-- evaluate.py
```

## Dataset Columns

- `Applicant_ID`
- `Applicant_Income`
- `Coapplicant_Income`
- `Employment_Status`
- `Age`
- `Marital_Status`
- `Dependents`
- `Credit_Score`
- `Existing_Loans`
- `DTI_Ratio`
- `Savings`
- `Collateral_Value`
- `Loan_Amount`
- `Loan_Term`
- `Loan_Purpose`
- `Property_Area`
- `Education_Level`
- `Gender`
- `Employer_Category`
- `Loan_Approved` (target)

## ML Workflow

1. Data loading and validation
2. Data preprocessing (missing values, encoding, scaling)
3. Baseline model training
4. Hyperparameter tuning
5. Evaluation and reporting
6. API inference

## Quick Start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run preprocessing:

```bash
python -m src.preprocess
```

4. Train baseline models:

```bash
python -m src.train_baseline
```

5. Tune best model:

```bash
python -m src.tune_model
```

6. Start API server:

```bash
uvicorn app:app --reload
```

7. Start frontend (Streamlit):

```bash
streamlit run streamlit_app.py
```

## API Endpoint

- `POST /predict`
- Request body: applicant feature values (except `Applicant_ID` and `Loan_Approved`)
- Response: predicted class and approval probability

## Frontend Demo

This project includes a Streamlit frontend:

- Run `streamlit run streamlit_app.py`
- Open the local URL shown in terminal (usually `http://localhost:8501`)
- Fill applicant details and click `Predict Loan Decision`

## Fairness and Explainability

Notebook `notebooks/02_explainability_bias.ipynb` includes:

- Group-level approval rate checks
- Group-level TPR/FPR checks
- Feature importance review

## Future Enhancements

- Add SHAP plots for per-applicant reasoning
- Add model registry and experiment tracking
- Add CI/CD and container deployment
