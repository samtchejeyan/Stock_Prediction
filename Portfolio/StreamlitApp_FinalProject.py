import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

from joblib import dump, load

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import NumpyDeserializer

from imblearn.pipeline import Pipeline as ImbPipeline
import shap

# ── Setup ─────────────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load saved X_train - same folder as this script
# On Streamlit Cloud: /mount/src/stock_prediction/Portfolio/X_train.csv
file_path = os.path.join(current_dir, 'X_train.csv')
dataset = pd.read_csv(file_path)
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

# ── AWS Secrets ───────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket   = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

# ── AWS Session ───────────────────────────────────────────────────────────────
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

# ── Model Configuration ───────────────────────────────────────────────────────
# Top 5 features from SHAP waterfall plot (int_rate and grade are strongest drivers)
MODEL_INFO = {
    "endpoint"  : aws_endpoint,
    "explainer" : "explainer_loan.shap",
    "pipeline"  : "finalized_loan_model.tar.gz",
    "keys"      : ['int_rate', 'grade', 'annual_inc', 'dti_ratio', 'loan_to_income'],
    "inputs"    : [
        {
            "name"    : "int_rate",
            "label"   : "Interest Rate (%)",
            "min"     : 5.0,
            "max"     : 30.0,
            "default" : 13.0,
            "step"    : 0.1,
            "help"    : "The loan interest rate assigned to the borrower."
        },
        {
            "name"    : "grade",
            "label"   : "Loan Grade  (0 = A  …  6 = G)",
            "min"     : 0.0,
            "max"     : 6.0,
            "default" : 2.0,
            "step"    : 1.0,
            "help"    : "LendingClub credit grade. A (0) is lowest risk, G (6) is highest."
        },
        {
            "name"    : "annual_inc",
            "label"   : "Annual Income — log scale  (e.g. 11 ≈ $60k)",
            "min"     : 8.0,
            "max"     : 14.0,
            "default" : 11.0,
            "step"    : 0.1,
            "help"    : "log1p(annual income). 11 ≈ $60k, 12 ≈ $162k, 10 ≈ $22k."
        },
        {
            "name"    : "dti_ratio",
            "label"   : "Debt-to-Income Ratio (monthly payment / income)",
            "min"     : 0.0,
            "max"     : 1.0,
            "default" : 0.005,
            "step"    : 0.001,
            "help"    : "Monthly installment divided by annual income. Higher = more financial stress."
        },
        {
            "name"    : "loan_to_income",
            "label"   : "Loan-to-Income Ratio (loan amount / income)",
            "min"     : 0.0,
            "max"     : 2.0,
            "default" : 0.15,
            "step"    : 0.01,
            "help"    : "Loan amount divided by annual income. Higher = greater repayment burden."
        },
    ]
}

# ── Load Pipeline from S3 ─────────────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{key}/{os.path.basename(filename)}"
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)

# ── Load SHAP Explainer from S3 ───────────────────────────────────────────────
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)

    with open(local_path, "rb") as f:
        return load(f)

# ── Prediction via SageMaker Endpoint ────────────────────────────────────────
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        # Convert DataFrame to dict for JSON serialization
        if isinstance(input_df, pd.DataFrame):
            input_df = input_df.to_dict(orient='list')
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        mapping  = {0: "✅ Fully Paid", 1: "⚠️ Likely Default"}
        return mapping.get(int(pred_val), str(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# ── SHAP Waterfall Plot ───────────────────────────────────────────────────────
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    local_path     = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer     = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        local_path
    )
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Run preprocessing steps only (exclude model = last step)
    preprocessing_pipeline = ImbPipeline(steps=best_pipeline.steps[:-1])

    input_df_transformed = preprocessing_pipeline.transform(pd.DataFrame(input_df))
    feature_names        = best_pipeline[:-1].get_feature_names_out()
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names)

    shap_values = explainer(input_df_transformed, check_additivity=False)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 1], show=False)  # class 1 = default
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_values[0, :, 1].values, index=shap_values[0, :, 1].feature_names)
        .abs()
        .idxmax()
    )
    st.info(f"**Key Driver:** The most influential factor in this prediction was **{top_feature}**.")

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Loan Default Predictor", layout="wide")
st.title("🏦 Loan Default Predictor")
st.markdown(
    "Enter key loan and borrower details below. "
    "The model will predict the likelihood of default and explain the key drivers of that decision."
)

with st.form("pred_form"):
    st.subheader("Loan Application Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['label'],
                min_value=float(inp['min']),
                max_value=float(inp['max']),
                value=float(inp['default']),
                step=float(inp['step']),
                help=inp['help']
            )

    submitted = st.form_submit_button("🔍 Run Prediction")

# Fill remaining columns from saved X_train row 0
original = dataset.iloc[0:1].to_dict(orient='records')[0]
original.update(user_inputs)
input_df = pd.DataFrame([original])

if submitted:
    with st.spinner("Running prediction..."):
        res, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", res)

        if "Default" in res:
            st.warning(
                "⚠️ This applicant is flagged as **high-risk**. "
                "Consider manual review, adjusted interest rates, or reduced loan amount."
            )
        else:
            st.success(
                "✅ This applicant is predicted to **fully repay** the loan. "
                "Standard approval process may proceed."
            )

        with st.spinner("Generating SHAP explanation..."):
            display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
