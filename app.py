import streamlit as st
import pandas as pd
import xgboost as xgb 
import numpy as np 
import os

# Get the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_model.json")
model = xgb.Booster()
model.load_model(MODEL_PATH)


st.set_page_config(page_title="Loan Approval Predictor", page_icon="💸")

st.title("Loan Approval Prediction App")
st.write("Fill in the form below.")

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])

    # Fix categorical formatting
    df["person_education"] = df["person_education"].str.replace(" ", "_")

    # Categories
    df_encoded = pd.get_dummies(df, columns=[
        'person_gender',
        'person_education',
        'person_home_ownership',
        'loan_intent',
        'previous_loan_defaults_on_file'
    ], dtype=int)

    # Add any missing dummy columns (to match training set)
    for col in all_training_columns:   # you define this list below
        if col not in df_encoded:
            df_encoded[col] = 0

    # Ensure correct column order
    df_encoded = df_encoded[all_training_columns]

    return df_encoded

all_training_columns = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "person_gender_female",
    "person_gender_male",
    "person_education_Associate",
    "person_education_Bachelor",
    "person_education_Doctorate",
    "person_education_High_School",
    "person_education_Master",
    "person_home_ownership_MORTGAGE",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_DEBTCONSOLIDATION",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
    "previous_loan_defaults_on_file_No",
    "previous_loan_defaults_on_file_Yes"
]

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
loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value = 0)
loan_percent_income = (st.number_input("What Percent of your Income is the Loan? (%)", min_value = 0)) / 100
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

if st.button("Predict Loan Approval"):
    processed=preprocess_input(input_data)
    dmatrix = xgb.DMatrix(processed)

    prob = float(model.predict(dmatrix)[0])
    prob_display = round(prob, 4)

    st.subheader("Prediction Result")
    st.write(f"Approval Probability: **{prob_display}**")

    if prob >= 0.914:
        st.success("Your loan is likely to be Approved ✔️")
    else:
        st.error("Your loan is likely to be Denied ❌")

