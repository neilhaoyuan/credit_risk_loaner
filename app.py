import streamlit as st
import pandas as pd
import xgboost as xgb 
import numpy as np 
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.json")
model = xgb.Booster()
model.load_model(MODEL_PATH)


st.set_page_config(page_title="Loan Approval Predictor: ", page_icon="💸")

st.title("💸 Loan Default Risk Prediction App")
st.write("Fill in the form below to assess default risk.")

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


# -----------------------
# Input fields
# -----------------------

Age = st.number_input("Age", min_value=18, value=30)
Income = st.number_input("Annual Income ($)", min_value=0, value=50_000)
MonthsEmployed = st.number_input("Months Employed", min_value=0, value=24)
LoanAmount = st.number_input("Loan Amount Requested ($)", min_value=0, value=10_000)
InterestRate = round(st.number_input("Loan Interest Rate (%)", min_value=0.0, step=0.01, value=10.0), 2)
DTIRatio = round(st.number_input("Debt-to-Income Ratio (as decimal, e.g., 0.3 for 30%)", min_value=0.0, max_value=1.0, step=0.01, value=0.3), 4)
CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
NumCreditLines = st.number_input("Number of Credit Lines", min_value=0, value=3)
LoanTerm = st.number_input("Loan Term (months)", min_value=1, value=36)

Education = st.selectbox("Highest Education", ["High School", "Associate's", "Bachelor's", "Master's", "Doctorate"])
EmploymentType = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
LoanPurpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Medical", "Other"])

HasMortgage = 1 if st.selectbox("Has Mortgage?", ["No", "Yes"]) == "Yes" else 0
HasDependents = 1 if st.selectbox("Has Dependents?", ["No", "Yes"]) == "Yes" else 0
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

##--------------------------------
##Adding risk tolerance/threshold input section here so we can give to user
st.markdown("----")
st.subheader("Risk Tolerance Settings")
st.write("Select your desired risk threshold based on your lending strategy.")
st.write("**Lower threshold** = More conservative (reject more loans to avoid defaults)")
st.write("**Higher threshold** = More aggressive (approve more loans but accept higher risk)")
st.write("")
st.write("**How it works:** The model predicts the probability that a loan will default.")
st.write("- If default probability >= threshold -> reject the loan")
st.write("- If default probability < threshold -> approve the loan")

threshold_choice = st.radio(
    "Select Risk Threshold:",
    ["Conservative (0.116) - KS-Optimized threshold, precise at predicting good loans but has little risk tolerance",
     "Balanced (0.5) - Standard threshold, balances catching defaults vs approving good loans",
     "Aggressive (0.914) - High-risk threshold, approves more loans but misses more defaults",
     "Custom - Set your own threshold value"], 
    index=0 
)

if "Conservative" in threshold_choice:
    threshold = 0.116
    risk_description = "Will reject any loan with >11.6% default probability. Catches most defaults but may reject some good customers."
elif "Balanced" in threshold_choice:
    threshold = 0.5
    risk_description = "Will reject any loan with >50% default probability. Standard risk-neutral approach."
elif "Aggressive" in threshold_choice:
    threshold = 0.914
    risk_description = "Will only reject loans with >91.4% default probability. Approves most loans but accepts higher default risk."
else:
    threshold = st.number_input(
        "Enter custom threshold value (between 0 and 1):",
        min_value=0.0,
        max_value=1.0,
        value=0.116,
        step=0.01,
        help="Lower threshold = More conservative (reject if default risk > threshold). Higher threshold = More aggressive (only reject very high-risk loans)."
    )
    risk_description = f"**Custom threshold:** Will reject loans with >{threshold:.1%} default probability"

st.info(risk_description)

if st.button("Predict Loan Risk"):

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

    # -----------------------
    # Model prediction
    # -----------------------

    processed = preprocess_input(input_data)
    dmatrix = xgb.DMatrix(processed)

    default_prob = float(model.predict(dmatrix)[0])

    st.subheader("Prediction Result")
    st.write(f"**Default Probability:** {default_prob:.4f} ({default_prob*100:.2f}%)")
    st.write(f"**Threshold Used:** {threshold:.3f}")

    if default_prob >= threshold:
        st.error("Your loan is likely to be Denied ❌")
    else:
        st.success("Your loan is likely to be Approved ✔️")