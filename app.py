import streamlit as st
import pandas as pd

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💸")

st.title("Loan Approval Prediction App")
st.write("Fill in the form below.")

# -----------------------
# Input fields
# -----------------------

person_age = st.number_input("Age", min_value=18, value=30)
person_gender = st.selectbox("Gender", ["Female", "Male", "Other"])
person_education = st.selectbox("Highest Education", ["None", "High School", "Associates", "Bachelor", "Master"])
person_income = st.number_input("Annual Income ($)", min_value = 0)
person_emp_exp = st.number_input("Number of Years Worked", min_value = 0)
person_home_ownership = st.selectbox("Home Ownership Type", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amount = st.number_input("Loan Amount Requested ($)", min_value = 0)
loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "MEDICAL", "EDUCATIONAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
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
    "loan_amount": loan_amount,
    "loan_intent": loan_intent,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": previous_loan_defaults_on_file
}
