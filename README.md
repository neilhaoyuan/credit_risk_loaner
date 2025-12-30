
# 💳 Loan Oracle – Credit Risk Assessment Platform

Loan Oracle is an end-to-end **credit risk modeling and decision support system** that estimates the probability of loan default using machine learning. It combines a robust backend modeling pipeline with an interactive Streamlit frontend designed for both **lenders** and **borrowers**.

The project is built with transparency and real-world lending logic in mind, making it easy to understand, extend, and demonstrate.

---
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-ML%20Model-006400" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Numerical-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-Optimization-8CAAE6?logo=scipy&logoColor=white" />
</p>

## What the project does

### 1. Prepare and clean loan application data

The backend pipeline begins with a structured loan dataset containing borrower demographics, credit characteristics, and loan attributes.

Key preprocessing steps include:

* Cleaning and formatting categorical variables
* Encoding categorical features using one-hot encoding
* Converting binary fields into numeric form
* Creating interaction features that reflect known credit risk relationships

The final dataset is fully numeric and ready for model training.

---

### 2. Engineer credit-risk features

In addition to core financial variables, the model incorporates domain-informed features such as:

* High debt-to-income combined with low credit score
* Employment stability and homeownership indicators

These features help the model capture known patterns in borrower risk while still allowing the algorithm to learn non-linear relationships.

---

### 3. Train a gradient-boosted credit model

The core model is trained using **XGBoost**, a gradient-boosted decision tree algorithm well-suited for structured financial data.

Model training includes:

* Stratified train/test splitting to handle class imbalance
* Cross-validation using ROC-AUC
* Early stopping to prevent overfitting
* Evaluation using industry-standard metrics such as ROC-AUC and the Gini coefficient

The trained model outputs a **probability of default** for each loan application.

---

### 4. Optimize decision thresholds

Because loan datasets are naturally imbalanced, multiple approval strategies are evaluated:

* Fixed probability thresholds
* KS-optimal thresholds
* Custom, lender-defined risk tolerance levels

This allows the model to support different underwriting strategies rather than forcing a single approval rule.

---

### 5. Deploy an interactive Streamlit frontend

The Streamlit application allows users to interact with the model in real time.

Two modes are supported:

* **Lender view**: assess applicants under configurable risk policies
* **Borrower view**: estimate approval likelihood and receive actionable feedback

The interface includes:

* Structured input forms
* Real-time default probability prediction
* Approval or denial decisions
* Human-readable suggestions to improve approval chances

---

## Technology stack

**Language**

* Python

**Backend / Modeling**

* Jupyter Notebook
* XGBoost
* scikit-learn
* NumPy
* Pandas
* SciPy

**Frontend**

* Streamlit

**Modeling Techniques**

* Gradient Boosted Trees
* Cross-validation
* Probability-based decisioning
* Credit risk metrics (ROC-AUC, Gini, KS)

---

## Project structure

```
Loan-Oracle/
│
├── app.py                      # Streamlit frontend
├── loan_model.json             # Trained XGBoost model
├── training_columns.json       # Feature alignment for deployment
├── backend_notebook.ipynb      # Data prep, training, evaluation
├── README.md                   # Project documentation
├── .streamlit/
│   └── config.toml             # Theme configuration
└── assets/
    ├── LoanOracleTitle.png
    └── loanOracle.png
```

---

## How to run the project

1. Install dependencies

```bash
pip install streamlit xgboost pandas numpy scipy scikit-learn
```

2. Run the Streamlit app

```bash
streamlit run app.py
```

3. Open the app in your browser and select **Lender** or **Borrower** mode.

---

## Design philosophy

Loan Oracle is designed to:

* Reflect real credit risk modeling practices
* Separate **risk prediction** from **lending policy**
* Remain interpretable and explainable
* Be easily extended with additional features, rules, or models

The goal is not just prediction accuracy, but **decision usefulness**.

---

## Disclaimer

This project uses synthetic data and is intended for educational and demonstration purposes only.
It does not constitute financial advice or a guarantee of loan approval or denial.

