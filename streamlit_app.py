import joblib
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR


MODEL_PATH = MODELS_DIR / "best_model.pkl"


def get_risk_band(approval_probability: float) -> str:
    if approval_probability >= 0.75:
        return "Low Risk"
    if approval_probability >= 0.50:
        return "Medium Risk"
    return "High Risk"


def is_manual_review_needed(approval_probability: float) -> bool:
    return 0.45 <= approval_probability <= 0.55


def get_eligibility_flags(applicant: dict) -> list[str]:
    flags = []
    if applicant["Credit_Score"] < 650:
        flags.append("Low credit score (< 650)")
    if applicant["DTI_Ratio"] > 45:
        flags.append("High DTI ratio (> 45%)")
    if applicant["Savings"] < 50000:
        flags.append("Low savings buffer (< 50,000)")
    if applicant["Existing_Loans"] >= 3:
        flags.append("Multiple running loans (>= 3)")
    if applicant["Collateral_Value"] < applicant["Loan_Amount"] * 0.8:
        flags.append("Weak collateral coverage (< 80% of loan amount)")
    return flags


def get_actionable_suggestions(applicant: dict, flags: list[str]) -> list[str]:
    suggestions = []
    if "Low credit score (< 650)" in flags:
        suggestions.append("Increase credit score above 650 by improving repayment history.")
    if "High DTI ratio (> 45%)" in flags:
        suggestions.append("Reduce debt burden so DTI ratio moves below 45%.")
    if "Low savings buffer (< 50,000)" in flags:
        suggestions.append("Build emergency savings to at least 50,000 before reapplying.")
    if "Multiple running loans (>= 3)" in flags:
        suggestions.append("Close or consolidate existing loans to reduce active loan count.")
    if "Weak collateral coverage (< 80% of loan amount)" in flags:
        suggestions.append("Add stronger collateral or request a smaller loan amount.")
    if not suggestions:
        suggestions.append("Applicant profile is healthy; proceed with standard document verification.")
    return suggestions


def estimate_top_factors(applicant: dict) -> list[tuple[str, str]]:
    total_income = applicant["Applicant_Income"] + applicant["Coapplicant_Income"]
    collateral_ratio = (
        applicant["Collateral_Value"] / applicant["Loan_Amount"]
        if applicant["Loan_Amount"] > 0
        else 0
    )

    factors = []
    factors.append(
        (
            "Credit_Score",
            "Positive impact" if applicant["Credit_Score"] >= 700 else "Negative impact",
        )
    )
    factors.append(
        ("DTI_Ratio", "Positive impact" if applicant["DTI_Ratio"] <= 40 else "Negative impact")
    )
    factors.append(
        ("Income Strength", "Positive impact" if total_income >= 60000 else "Negative impact")
    )
    factors.append(
        (
            "Collateral Coverage",
            "Positive impact" if collateral_ratio >= 1.0 else "Negative impact",
        )
    )
    factors.append(
        (
            "Existing_Loans",
            "Positive impact" if applicant["Existing_Loans"] <= 1 else "Negative impact",
        )
    )

    return factors


def factor_score_map(applicant: dict) -> dict[str, float]:
    total_income = applicant["Applicant_Income"] + applicant["Coapplicant_Income"]
    collateral_ratio = (
        applicant["Collateral_Value"] / applicant["Loan_Amount"]
        if applicant["Loan_Amount"] > 0
        else 0
    )
    return {
        "Credit Score": min(max((applicant["Credit_Score"] - 300) / 600, 0), 1),
        "DTI Health": min(max((60 - applicant["DTI_Ratio"]) / 60, 0), 1),
        "Income Strength": min(max(total_income / 150000, 0), 1),
        "Collateral Cover": min(max(collateral_ratio / 1.5, 0), 1),
        "Loan Burden": min(max((4 - applicant["Existing_Loans"]) / 4, 0), 1),
    }

st.set_page_config(page_title="CreditWise Loan Predictor", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 1280px;}
.hero {
    background: linear-gradient(135deg, #0f172a, #1e293b 55%, #334155);
    border-radius: 16px;
    padding: 20px 24px;
    border: 1px solid #334155;
    margin-bottom: 14px;
}
.hero h1 {
    margin: 0;
    font-size: 2rem;
    line-height: 1.2;
    color: #f8fafc;
    white-space: normal;
}
.hero p {margin: 6px 0 0 0; color: #cbd5e1;}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #0f172a;
}
.stMetric {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 10px;
}
[data-testid="stMetricLabel"] {
    color: #334155 !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-weight: 700 !important;
}
.result-card {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 14px;
    background: #ffffff;
    margin-bottom: 10px;
}
.accelerator-wrap {
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 14px 14px 12px 14px;
    background: #ffffff;
    margin-top: 10px;
}
.accelerator-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #475569;
    margin-top: 8px;
}
.accelerator-track {
    position: relative;
    width: 100%;
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #22c55e 100%);
    overflow: hidden;
}
.accelerator-indicator {
    position: absolute;
    top: -5px;
    width: 4px;
    height: 22px;
    border-radius: 2px;
    background: #0f172a;
    left: 1%;
    animation: drive 900ms ease-out forwards;
}
@keyframes drive { from { left: 1%; } to { left: var(--target); } }
.status-approved { color: #166534; font-weight: 700; }
.status-rejected { color: #b91c1c; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="hero">
    <h1>CreditWise Loan Approval Dashboard</h1>
    <p>Fast and consistent loan pre-screening with model-backed risk insights and recommendation support.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run: python -m src.preprocess, python -m src.train_baseline, python -m src.tune_model"
    )
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Applicant Details")
with st.container(border=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Applicant Profile")
        age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
        employment_status = st.selectbox(
            "Employment", ["Salaried", "Self-Employed", "Business"]
        )
        employer_category = st.selectbox("Employer Category", ["Govt", "Private", "Self"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1, step=1)
        education_level = st.selectbox(
            "Education", ["Graduate", "Postgraduate", "Undergraduate"]
        )

    with col2:
        st.markdown("#### Income and Debt")
        applicant_income = st.number_input(
            "Applicant Income", min_value=0.0, value=60000.0, step=1000.0
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income", min_value=0.0, value=10000.0, step=1000.0
        )
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=720, step=1)
        dti_ratio = st.slider(
            "DTI Ratio (%)", min_value=0.0, max_value=100.0, value=35.0, step=0.5
        )
        existing_loans = st.slider("Existing Loans", min_value=0, max_value=20, value=1, step=1)
        savings = st.number_input("Savings", min_value=0.0, value=150000.0, step=5000.0)

    with col3:
        st.markdown("#### Loan Request Details")
        loan_amount = st.number_input(
            "Loan Amount", min_value=1000.0, value=300000.0, step=10000.0
        )
        collateral_value = st.number_input(
            "Collateral Value", min_value=0.0, value=500000.0, step=10000.0
        )
        loan_term = st.slider("Loan Term (months)", min_value=6, max_value=480, value=180, step=6)
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Education", "Personal", "Business"])
        property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
        gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict Loan Decision", type="primary"):
    applicant_payload = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }
    input_df = pd.DataFrame([applicant_payload])

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    decision = "Approved" if prediction == 1 else "Rejected"
    risk_band = get_risk_band(probability)
    manual_review = is_manual_review_needed(probability)
    flags = get_eligibility_flags(applicant_payload)
    suggestions = get_actionable_suggestions(applicant_payload, flags)
    top_factors = estimate_top_factors(applicant_payload)
    factor_scores = factor_score_map(applicant_payload)

    st.markdown("---")
    st.subheader("Prediction Result")
    if decision == "Approved":
        st.markdown(
            f'<div class="result-card">Decision: <span class="status-approved">{decision}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-card">Decision: <span class="status-rejected">{decision}</span></div>',
            unsafe_allow_html=True,
        )

    result_col1, result_col2, result_col3 = st.columns(3)
    with result_col1:
        st.metric("Approval Probability", f"{probability * 100:.2f}%")
    with result_col2:
        st.metric("Risk Band", risk_band)
    with result_col3:
        st.metric("Prediction Class", str(prediction))

    accelerator_position = int(min(max(probability, 0.0), 1.0) * 96 + 2)
    st.markdown(
        f"""
<div class="accelerator-wrap">
    <div class="section-title" style="margin-bottom:10px;">Risk Accelerator</div>
    <div class="accelerator-track">
        <div class="accelerator-indicator" style="--target:{accelerator_position}%;"></div>
    </div>
    <div class="accelerator-labels">
        <span>High Risk</span>
        <span>Medium Risk</span>
        <span>Low Risk</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if manual_review:
        st.warning(
            "Confidence warning: probability is near 50%. Marking this case as Manual Review Recommended."
        )

    viz_col1, viz_col2 = st.columns([1, 1])
    with viz_col1:
        st.subheader("Top Factors (Influence)")
        factor_df = pd.DataFrame(
            [{"Factor": name, "Impact": effect} for name, effect in top_factors]
        )
        st.dataframe(factor_df, hide_index=True, use_container_width=True)

    with viz_col2:
        st.subheader("Factor Score Graph")
        score_df = pd.DataFrame(
            {"Factor": list(factor_scores.keys()), "Score": list(factor_scores.values())}
        ).set_index("Factor")
        st.bar_chart(score_df)

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.subheader("Eligibility Checks")
        if flags:
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.markdown("- No major eligibility flags triggered.")

    with detail_col2:
        st.subheader("Actionable Suggestions")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")
