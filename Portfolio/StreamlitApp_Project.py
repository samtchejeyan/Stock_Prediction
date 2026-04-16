import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

# Setup & Path Configuration
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features_loan

# Access the secrets
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# AWS Session Management
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
df_features = extract_features_loan()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "pipeline": "finalized_loan_model.tar.gz",
    "keys": [
        "loan_amnt", "term", "int_rate", "installment",
        "grade", "emp_length", "home_ownership", "annual_inc", "purpose"
    ],
    "inputs": [
        {"name": "loan_amnt",   "label": "Loan Amount ($)",         "min": 500.0,   "max": 40000.0,  "default": 10000.0, "step": 500.0},
        {"name": "int_rate",    "label": "Interest Rate (%)",        "min": 5.0,     "max": 30.0,     "default": 13.0,    "step": 0.5},
        {"name": "installment", "label": "Monthly Installment ($)",  "min": 50.0,    "max": 1500.0,   "default": 300.0,   "step": 10.0},
        {"name": "annual_inc",  "label": "Annual Income ($)",        "min": 10000.0, "max": 500000.0, "default": 60000.0, "step": 1000.0},
    ],
    "selects": [
        {"name": "term",           "label": "Loan Term (months)",        "options": [36, 60],
         "default": 36},
        {"name": "grade",          "label": "Loan Grade",                "options": ["A","B","C","D","E","F","G"],
         "default": "B"},
        {"name": "home_ownership", "label": "Home Ownership",            "options": ["MORTGAGE","OWN","RENT","OTHER"],
         "default": "MORTGAGE"},
        {"name": "emp_length",     "label": "Employment Length (years)", "options": [
             "< 1 year","1 year","2 years","3 years","4 years",
             "5 years","6 years","7 years","8 years","9 years","10+ years"],
         "default": "5 years"},
        {"name": "purpose",        "label": "Loan Purpose",              "options": [
             "debt_consolidation","credit_card","home_improvement","other",
             "major_purchase","medical","small_business","car",
             "vacation","moving","house","educational"],
         "default": "debt_consolidation"},
    ]
}

# Encoding maps — must match label encoding used during training
GRADE_MAP     = {g: i for i, g in enumerate(sorted(["A","B","C","D","E","F","G"]))}
OWNERSHIP_MAP = {o: i for i, o in enumerate(sorted(["MORTGAGE","OTHER","OWN","RENT"]))}
PURPOSE_MAP   = {p: i for i, p in enumerate(sorted([
    "car","credit_card","debt_consolidation","educational",
    "home_improvement","house","major_purchase","medical",
    "moving","other","small_business","vacation"
]))}
EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2,  "3 years": 3,
    "4 years":  4, "5 years": 5, "6 years": 6,  "7 years": 7,
    "8 years":  8, "9 years": 9, "10+ years": 10
}


# Prediction Logic
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df.values.astype(float))
        pred_val = int(pd.DataFrame(raw_pred).values[-1][0])
        mapping  = {0: "✅ Fully Paid", 1: "⚠️ Charged Off (Default)"}
        return mapping.get(pred_val, str(pred_val)), pred_val, 200
    except Exception as e:
        return f"Error: {str(e)}", None, 500


# Streamlit UI
st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("🏦 Loan Default Predictor")
st.markdown(
    "Enter borrower details below to predict whether a loan is likely to be "
    "**fully paid** or **charged off (defaulted)**."
)

with st.form("pred_form"):
    st.subheader("Loan & Borrower Inputs")
    cols = st.columns(2)

    user_inputs = {}

    # Number inputs
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["label"],
                min_value=inp["min"],
                max_value=inp["max"],
                value=float(inp["default"]),
                step=inp["step"]
            )

    # Select inputs
    for i, sel in enumerate(MODEL_INFO["selects"]):
        with cols[1]:
            user_inputs[sel["name"]] = st.selectbox(
                sel["label"],
                options=sel["options"],
                index=sel["options"].index(sel["default"])
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:

    # Encode categorical inputs to match training label encoding
    encoded = {
        "loan_amnt":      user_inputs["loan_amnt"],
        "term":           float(user_inputs["term"]),
        "int_rate":       user_inputs["int_rate"],
        "installment":    user_inputs["installment"],
        "grade":          GRADE_MAP[user_inputs["grade"]],
        "emp_length":     EMP_MAP[user_inputs["emp_length"]],
        "home_ownership": OWNERSHIP_MAP[user_inputs["home_ownership"]],
        "annual_inc":     np.log1p(user_inputs["annual_inc"]),
        "purpose":        PURPOSE_MAP[user_inputs["purpose"]],
    }

    input_df = pd.DataFrame([encoded], columns=MODEL_INFO["keys"])

    result, pred_val, status = call_model_api(input_df)

    if status == 200:
        st.divider()
        st.subheader("Prediction Result")
        st.metric("Outcome", result)

        if pred_val == 1:
            st.error(
                "**Business Insight:** This borrower is predicted to **default**. "
                "Lenders should consider adjusting the interest rate, reducing the loan amount, "
                "or requiring additional collateral before approving this application."
            )
        else:
            st.success(
                "**Business Insight:** This borrower is predicted to **fully repay** the loan. "
                "Based on the provided inputs, the model identifies this as a low-risk application."
            )
    else:
        st.error(result)
