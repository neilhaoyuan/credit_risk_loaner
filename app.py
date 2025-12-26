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

st.title("💸Loan Approval Prediction App")
st.write("Fill in the form below.")

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])

    df["person_education"] = df["person_education"].str.replace(" ", "_")

    df_encoded = pd.get_dummies(df, columns=[
        'person_gender',
        'person_education',
        'person_home_ownership',
        'loan_intent',
        'previous_loan_defaults_on_file'
    ], dtype=int)

    #matching the TRAINING cuz different, Rename to dti_ratio
    df_encoded.rename(columns={'loan_percent_income': 'dti_ratio'}, inplace=True)

    dti_ratio_preference = 0.4
    credit_score_preference = 600

    df_encoded['high_dti_low_credit'] = (
        (df_encoded['dti_ratio'] > dti_ratio_preference) & 
        (df_encoded['credit_score'] < credit_score_preference)
    ).astype(int)

    df_encoded['stable_homeowner'] = (
        (df_encoded.get('person_home_ownership_OWN', 0) == 1) & 
        (df_encoded['person_emp_exp'] > 5)
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

person_age = st.number_input("Age", min_value=18, value=30)
person_gender = st.selectbox("Gender", ["female", "male"])
person_education = st.selectbox("Highest Education", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
person_income = st.number_input("Annual Income ($)", min_value = 0)
person_emp_exp = st.number_input("Number of Years Worked", min_value = 0)
person_home_ownership = st.selectbox("Home Ownership Type", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Loan Amount Requested ($)", min_value = 0)
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "MEDICAL", "EDUCATION", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_int_rate = round(st.number_input("Loan Interest Rate (%)", min_value = 0.0, step=0.01),2)
loan_percent_income = round(st.number_input("What Percent of your Income is the Loan? (%)", min_value = 0.0, step=0.01)/100, 4)
cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults on File?", ["No", "Yes"])

input_data = {
    "person_age": person_age,
    "person_gender": person_gender,
    "person_education": person_education,
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "person_home_ownership": person_home_ownership,
    "loan_amnt": loan_amnt,
    "loan_intent": loan_intent,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": previous_loan_defaults_on_file
}

##--------------------------------
##Adding risk tolerance/threshold input section here so we can give to user
st.markdown("----")
st.subheader("🚨🎯Risk Tolerance Settings🎯🚨")
st.write("Select you desired approval threshold, based on your risk tolerance.")
st.write("Higher threshold means stricter approval criteria (lower risk tolerance).")
st.write("This is essentially a confidence level for the model, what is the rate of confidence you want the model to have, for it to approve the loan?")
st.write("For example, if you select 0.914, the model will only approve loans where it is at least 91.4% confident that the loan should be approved.")
st.write("You can adjust this threshold based on your risk tolerance, and values")

threshold_choice = st.radio(
    "Select Approval Threshold:",
    ["Aggresive (0.198) - KS Ratio Optimized threshold, which will approve more loans but with higher risk",
     "Balanced (0.5) - Neutral threshold, optimized for balanced approvals and denials, and quality results with both",
     "Conservative (0.914) - High confidence threshold, which will approve fewer loans but with lower risk",
     "Custom - Set your own threshold value"], index=2 #Default to Conservative
)

if "Aggresive" in threshold_choice:
    threshold = 0.198
    risk_description = "Less risk averse, prefer model to be only 19.8% confident to approve"
elif "Balanced" in threshold_choice:
    threshold = 0.5
    risk_description = "Neutral risk tolerance, prefer model to be at least 50% confident to approve"
elif "Conservative" in threshold_choice:
    threshold = 0.914
    risk_description = "More risk averse, prefer model to be at least 91.4% confident to approve"
else:
    threshold = st.number_input(
        "Enter custom threshold value (between 0 and 1):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Lower values(Less risk averse) = you are okay with model being less confident to approve loans, Higher values(More risk averse) = you want model to be more confident to approve loans"
    )
    risk_description = f"**Custom threshold**: Using threshold of {threshold:.3f}"

st.info(risk_description)

if st.button("Predict Loan Approval"):


    if person_age < 18:
        st.error("Applicant must be at least 18 years old.")
        st.stop()

    if person_income <= 0:
        st.error("Annual income must be greater than $0.")
        st.stop()

    if loan_amnt <= 0:
        st.error("Loan amount must be greater than $0.")
        st.stop()

    if not (300 <= credit_score <= 900):
        st.error("Credit score must be between 300 and 900.")
        st.stop()

    if loan_percent_income <= 0 or loan_percent_income > 1:
        st.error("Loan percent of income must be between 0% and 100%.")
        st.stop()

    # -----------------------
    # Model prediction
    # -----------------------

    processed = preprocess_input(input_data)
    dmatrix = xgb.DMatrix(processed)

    prob = float(model.predict(dmatrix)[0])

    st.subheader("Prediction Result")
    st.write(f"**Approval Probability:** {prob:.4f} ({prob*100:.2f}%)")
    st.write(f"**Threshold Used:** {threshold:.3f}")

    if prob >= threshold:
        st.success("Your loan is likely to be Approved ✔️")
    else:
        st.error("Your loan is likely to be Denied ❌")