import streamlit as st
import pandas as pd
import xgboost as xgb 
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.json")

model = xgb.Booster()
model.load_model(MODEL_PATH)

st.set_page_config(
    page_title="Loan Oracle",
    page_icon="loanOracle.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.block-container {
    max-width: 1200px;
}
</style>
""", unsafe_allow_html=True)

st.image("LoanOracleTitle.png", use_container_width=True)

st.markdown("---")

# User type selection
col1, col2 = st.columns(2)
with col1:
    if st.button("I'm a Lender", use_container_width=True, type="primary"):
        st.session_state.user_type = "lender"
with col2:
    if st.button("I'm a Borrower", use_container_width=True, type="primary"):
        st.session_state.user_type = "borrower"

# Initialize user_type if not set
if "user_type" not in st.session_state:
    st.session_state.user_type = None

# Show selection prompt or continue
if st.session_state.user_type is None:
    st.info("Please select whether you're a lender or a borrower to continue.")
    st.stop()
else:
    user_type = st.session_state.user_type
    if user_type == "lender":
        st.success("**Lender View:** Assess applicants under your risk strategy.")
    else:
        st.success("**Borrower View:** Estimate your loan approval likelihood.")

st.markdown("---")

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    df_encoded = pd.get_dummies(df, columns=[
        'Education',
        'EmploymentType',
        'MaritalStatus',
        'LoanPurpose',
    ], dtype=int)
    
    dti_ratio_preference = 0.4
    credit_score_preference = 600
    
    df_encoded['high_dti_low_credit'] = (
        (df_encoded['DTIRatio'] > dti_ratio_preference) & 
        (df_encoded['CreditScore'] < credit_score_preference)
    ).astype(int)
    
    df_encoded['stable_homeowner'] = (
        (df_encoded['HasMortgage'] == 1) & 
        (df_encoded['MonthsEmployed'] > 60)
    ).astype(int)
    
    for col in all_training_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[all_training_columns]
    return df_encoded

COLUMNS_PATH = os.path.join(BASE_DIR, "training_columns.json")
with open(COLUMNS_PATH, "r") as f:
    all_training_columns = json.load(f)

# Input fields
left, right = st.columns([1,1])

with left:
    st.subheader("Borrower Profile" if user_type == "lender" else "Your Profile")
    st.caption("Demographic background")
    Age = st.number_input("Age", min_value=18, value=30)
    Income = st.number_input("Annual Income ($)", min_value=0, value=50_000, help="Total annual income before taxes")
    MonthsEmployed = st.number_input("Months Employed", min_value=0, value=24)
    EmploymentType = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    Education = st.selectbox("Highest Education", ["High School", "Associate's", "Bachelor's", "Master's", "Doctorate"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    HasDependents = 1 if st.selectbox("Has Dependents?", ["No", "Yes"]) == "Yes" else 0
    HasMortgage = 1 if st.selectbox("Has Mortgage?", ["No", "Yes"]) == "Yes" else 0

with right:
    st.subheader("")
    st.caption("Credit quality and loan structure")
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650, help="FICO score (between 300-900)")
    DTIRatio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, step=0.01, value=0.3, help="Total monthly debt / gross monthly income (as a decimal)")
    NumCreditLines = st.number_input("Number of Credit Lines", min_value=0, value=3, help="Number of active credit accounts")
    LoanAmount = st.number_input("Loan Amount Requested ($)", min_value=0, value=10_000)
    LoanTerm = st.number_input("Loan Term (months)", min_value=1, value=36, help="Repayment period in months")
    InterestRate = st.number_input("Loan Interest Rate (%)", min_value=0.0, step=0.01, value=10.0)
    LoanPurpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Medical", "Other"])
    HasCoSigner = 1 if st.selectbox("Has Co-Signer?", ["No", "Yes"]) == "Yes" else 0

input_data = {
    "Age": Age,
    "Income": Income,
    "MonthsEmployed": MonthsEmployed,
    "LoanAmount": LoanAmount,
    "InterestRate": InterestRate,
    "DTIRatio": DTIRatio,
    "CreditScore": CreditScore,
    "NumCreditLines": NumCreditLines,
    "LoanTerm": LoanTerm,
    "Education": Education,
    "EmploymentType": EmploymentType,
    "MaritalStatus": MaritalStatus,
    "LoanPurpose": LoanPurpose,
    "HasMortgage": HasMortgage,
    "HasDependents": HasDependents,
    "HasCoSigner": HasCoSigner
}

st.markdown("---")

# Risk Strategy
st.subheader("Risk Strategy" if user_type == "lender" else "Your Lender's Risk Tolerance")
st.caption("Adjust approval aggressiveness based on your preferences" if user_type == "lender" else "Adjust the approval threshold based on your lender's expected preferences")

risk_profile = st.radio(
    "Risk Profile",
    ["Conservative", "Balanced", "Aggressive", "Custom"],
    horizontal=True,
)

if risk_profile == "Conservative":
    threshold = 0.116
    profile_desc = "**Conservative:** Prioritizes capital preservation. Only accepts applicants with very low risk."
elif risk_profile == "Balanced":
    threshold = 0.35
    profile_desc = "**Balanced**: Balances approval volume with default protection."
elif risk_profile == "Aggressive":
    threshold = 0.55
    profile_desc = "**Aggressive**: Maximizes approvals while accepting higher default risk."
else:
    threshold = st.slider(
        "Custom Risk Threshold",
        min_value=0.001,
        max_value=0.999,
        value=0.420,
        step=0.01,
        help="Applications with probability of default below this threshold will be approved."
    )
    profile_desc = f"**Custom:** Manually defined approval threshold at {threshold:.1%}."

st.info(profile_desc)

st.markdown("---")

if st.button("🔍 Predict Loan Risk" if user_type == "lender" else "🔍 Check My Approval Chances",
             use_container_width=True,
             type="primary"):
    # Validation
    if Age < 18:
        st.error("Applicant must be at least 18 years old.")
        st.stop()
    if Income <= 0:
        st.error("Annual income must be greater than $0.")
        st.stop()
    if LoanAmount <= 0:
        st.error("Loan amount must be greater than $0.")
        st.stop()
    if not (300 <= CreditScore <= 900):
        st.error("Credit score must be between 300 and 900.")
        st.stop()
    if DTIRatio <= 0 or DTIRatio > 1:
        st.error("DTI Ratio must be between 0 and 1.")
        st.stop()
    
    # Model prediction
    processed = preprocess_input(input_data)
    dmatrix = xgb.DMatrix(processed)
    default_prob = float(model.predict(dmatrix)[0])
    
    st.markdown("---")
    st.subheader("Prediction Results")

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.metric("Default Probability", f"{default_prob*100:.2f}%")
    with stat2: 
        st.metric("Risk Threshold", f"{threshold*100:.2f}%")
    with stat3: 
        decision = "Approved" if default_prob < threshold else "Denied"
        st.metric("Decision",decision)
    
    if default_prob >= threshold:
        if user_type == "lender":
            st.error("### ❌ Loan Recommended for Denial")
            st.write(
                f"The predicted default probability (**{default_prob*100:.2f}%**) exceeds your selected risk threshold (**{threshold*100:.2f}%**). Approving this loan would violate your current risk policy.")
        else:
            st.error("### ❌ Loan Likely to be Denied")
            st.write(
                f"Your estimated default probability (**{default_prob*100:.2f}%**) exceeds the lender's risk threshold (**{threshold*100:.2f}%**).")
            st.markdown("**Suggestions to improve approval chances:**")
            suggestions = []
            if CreditScore < 650:
                suggestions.append("- Improve your credit score by paying bills on time and reducing your credit use")
            if DTIRatio > 0.4:
                suggestions.append("- Lower your debt-to-income ratio by paying down your existing debts")
            if MonthsEmployed < 24:
                suggestions.append("- Build a greater employment history for better stability")
            if not HasCoSigner and LoanAmount > Income * 0.3:
                suggestions.append("- Consider adding a co-signer to strengthen the application")
            
            if suggestions:
                for suggestion in suggestions:
                    st.write(suggestion)
            else:
                st.write("- Consider applying with a different lender or adjusting loan terms")
    else:
        if user_type == "lender":
            st.success("### ✅ Loan Recommended for Approval")
            st.write(
                f"The predicted default probability (**{default_prob*100:.2f}%**) is below your selected risk threshold (**{threshold*100:.2f}%**). This loan meets your risk criteria.")
        else:
            st.success("### ✅ Loan Likely to be Approved")
            st.write(
                f"Your estimated default probability (**{default_prob*100:.2f}%**) falls within the lender's acceptable risk range (below **{threshold*100:.2f}%**).")
            st.write("Your application meets the lender's criteria based on the provided information.")


st.markdown("---")

st.caption("**Disclaimer:** This tool provides risk estimates based on our machine learning model built on 250,000 cases of synthetic data and should not be considered as financial advice or guarantee of a loan approval/denial. Actual lending decisions require additional factors and human judgement. Please consult with professionals for proper guidance.")