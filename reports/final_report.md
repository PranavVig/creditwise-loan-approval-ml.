# Final Project Report

## Objective

Build a machine learning system that predicts loan approval decisions to reduce manual verification time and improve decision consistency.

## Pipeline

1. Load and validate historical loan application data.
2. Split data into train and test sets.
3. Build preprocessing for numerical and categorical features.
4. Train baseline classifiers.
5. Tune selected model.
6. Evaluate performance and fairness indicators.
7. Expose prediction API for operational use.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Fairness Checks

The project includes grouped analysis for:

- Gender
- Property Area

These checks help identify disparate approval rates and error rates.

## Deployment Readiness

- Model artifact saved in `models/best_model.pkl`
- Prediction service available in `app.py` via FastAPI
- Reproducible scripts under `src/`

## Recommended Next Steps

- Integrate with production loan workflow
- Add monitoring for drift and fairness
- Retrain regularly using latest approved/rejected outcomes
