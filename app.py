import streamlit as st
import pandas as pd
import xgboost as xgb 
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.json")

model = xgb.Booster()
model.load_model(MODEL_PATH)

# Sets up the page bar in the tabs list 
st.set_page_config(
    page_title="Loan Oracle",
    page_icon=os.path.join(BASE_DIR, "assets", "loanOracle.png"),
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

# The logo at the start
st.image(os.path.join(BASE_DIR, "assets", "LoanOracleTitle.png"), use_container_width=True)

st.markdown("---") # These create page breaks (I love them)

# User type selection (Lender vs Borrower)
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

# Explain what they've chosen (or tell them to choose)
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

COLUMNS_PATH = os.path.join(BASE_DIR, "model", "training_columns.json")
with open(COLUMNS_PATH, "r") as f:
    all_training_columns = json.load(f)

# Input fields
left, right = st.columns([1,1])

DTIRatio = 0

with left:
    st.subheader("Borrower Profile" if user_type == "lender" else "Your Profile")
    st.caption("Demographic background")
    Age = st.number_input("Age", min_value=18, value=30)
    Income = st.number_input("Annual Income After Taxes ($)", min_value=0, value=50_000) * (1 - DTIRatio)
    MonthsEmployed = (st.number_input("Years Employed", min_value=0, value=2)) * 12
    EmploymentType = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    Education = st.selectbox("Highest Education", ["High School Diploma", "College Diploma", "Associate's", "Bachelor's", "Master's", "Doctorate"])
    if Education == "College Diploma":
        Education = "Associate's"
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
    LoanPurpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
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
    ["Conservative", "Balanced", "Aggressive", "KS Optimal", "Custom"],
    horizontal=True,
    help="Risk scores rank loans by relative default risk, they are not exact probabilities.  \nA 40% score does not mean a 40% chance of default."
)

if risk_profile == "Conservative":
    threshold = 0.2241
    profile_desc = "**Conservative (~22%):** Prioritizes capital preservation. Only accepts applicants with very low risk."
elif risk_profile == "Balanced":
    threshold = 0.3262
    profile_desc = "**Balanced (~33%)**: Balances approval volume with default protection. Optimizes f1 score."
elif risk_profile == "Aggressive":
    threshold = 0.4264
    profile_desc = "**Aggressive (~43%)**: Maximizes approvals while accepting higher default risk."
elif risk_profile == "KS Optimal":
    threshold = 0.49979624
    profile_desc = "**KS Optimal (~50%)**: Maximizes seperation between defaulters and non-defaulters using Kolmogorov-Smirnov statistic."
else:
    threshold = st.slider(
        "Custom Risk Threshold",        min_value=0.001,
        max_value=0.999,
        value=0.60,
        step=0.01,
        help="Applications with probability of default below this threshold will be approved."
    )
    profile_desc = f"**Custom:** Manually defined approval threshold at {threshold:.1%}."

st.info(profile_desc)

st.markdown("---")

# Actual model use done here and loan prediction outputs

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
    if DTIRatio < 0 or DTIRatio > 1:
        st.error("DTI Ratio must be between 0 and 1.")
        st.stop()
    
    # Model prediction
    processed = preprocess_input(input_data)
    dmatrix = xgb.DMatrix(processed)
    default_prob = float(model.predict(dmatrix)[0])
    
    monthly_payment = (LoanAmount * (InterestRate/100/12)) / (1 - (1 + InterestRate/100/12)**(-LoanTerm))
    monthly_income = Income / 12

    base_prob = default_prob
    reject_mult = 1
    accept_mult = 1
    # Reject mult
    if LoanPurpose != "Home":
        if  LoanAmount / Income > 2:
            reject_mult *= 5
        elif monthly_payment > monthly_income * 0.4:
            reject_mult *= 5.5
    else:
        if (LoanAmount / Income > 10):
            reject_mult *= 6
    if Age + (LoanTerm / 12) > 75:
        reject_mult *= 3
    # Accept mult
    if LoanAmount / Income < 0.1:
        accept_mult *= 0.5
    if monthly_payment < monthly_income * 0.1:
        accept_mult *= 0.5

    default_prob = min(base_prob*reject_mult*accept_mult, 1.0)

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
                suggestions.append("- Improve your credit score by paying bills on time and reducing your credit use.")
            if DTIRatio > 0.4:
                suggestions.append("- Lower your debt-to-income ratio by paying down your existing debts.")
            if MonthsEmployed < 24:
                suggestions.append("- Build a greater employment history for better stability.")
            if not HasCoSigner and LoanAmount > Income * 0.3:
                suggestions.append("- Consider adding a co-signer to strengthen the application.")
            if LoanPurpose == 'Home':
                if LoanAmount / Income > 10:
                    suggestions.append("- Home loan exceeds 10x annual income. Consider raising your income or looking for a less expensive alternative.")
            else: 
                if LoanAmount / Income > 2:
                    suggestions.append("- Loan amount exceeds 2x annual income. Consider raising your income or looking for a less expensive alternative.")
                    
            if monthly_payment > monthly_income * 0.5:
                suggestions.append("- Monthly payment would exceed 40% of gross income. Increase your income, negotiate lower interest, or find a less expensive alternative.")

            if Age + (LoanTerm / 12) > 75:
                suggestions.append("- Loan term extends significantly into retirement years.")
            if suggestions:
                for suggestion in suggestions:
                    st.write(suggestion)
            else:
                st.write("- Consider applying with a different lender or adjusting loan terms.")
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

#Literally just a disclaimer at the bottom to not use as financial advice
st.caption("**Disclaimer:** This tool provides risk estimates based on our machine learning model trained on 60,000 balanced credit profiles (F1-score: 0.69). This should not be considered as financial advice or guarantee of loan approval/denial. Actual lending decisions require additional factors and human judgment. Consult with financial professionals for proper guidance.")
st.markdown("---")
st.caption("If you wanna check out the project, try: https://github.com/neilhaoyuan/credit_risk_loaner/tree/main")