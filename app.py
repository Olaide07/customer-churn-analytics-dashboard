import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Telco_customer_churn.xlsx")
    return df

df = load_data()

# -------------------------------
# PREPROCESSING
# -------------------------------
def preprocess(df):
    df = df.copy()

    drop_cols = [
        "CustomerID", "Count", "Country", "State", "City",
        "Zip Code", "Lat Long", "Latitude", "Longitude",
        "Churn Label", "Churn Score", "CLTV", "Churn Reason"
    ]

    df = df.drop(columns=drop_cols, errors="ignore")

    # Fix Total Charges
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Total Charges"].fillna(df["Total Charges"].median(), inplace=True)

    # One-hot encoding
    df_encoded = pd.get_dummies(df)

    return df_encoded

df_encoded = preprocess(df)

# -------------------------------
# ALIGN FEATURES WITH MODEL (CRITICAL FIX)
# -------------------------------
model_features = model.get_booster().feature_names

# Add missing columns
for col in model_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Keep only model columns in correct order
df_encoded = df_encoded[model_features]

# -------------------------------
# SIDEBAR CONTROLS
# -------------------------------
st.sidebar.title("⚙️ Controls")

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.4)

st.sidebar.markdown("### 🔎 Filters")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

payment_filter = st.sidebar.multiselect(
    "Payment Method",
    options=df["Payment Method"].unique(),
    default=df["Payment Method"].unique()
)

# -------------------------------
# FILTER DATA
# -------------------------------
filtered_df = df[
    (df["Contract"].isin(contract_filter)) &
    (df["Payment Method"].isin(payment_filter))
]

# -------------------------------
# PREDICTION (FINAL FIX)
# -------------------------------
X = df_encoded

# Use feature names to avoid mismatch
dmatrix = xgb.DMatrix(X, feature_names=model_features)

# Use booster (NOT sklearn wrapper)
y_prob = model.get_booster().predict(dmatrix)

y_pred = (y_prob >= threshold).astype(int)

# -------------------------------
# DASHBOARD UI
# -------------------------------
st.title("📊 Customer Churn Analytics Dashboard")
st.markdown("### Built by Olaide Ajibade")

# Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Actual Churn Rate", f"{df['Churn Value'].mean()*100:.2f}%")
col3.metric("Predicted Churn", int(y_pred.sum()))

st.markdown("---")

# -------------------------------
# CHARTS
# -------------------------------
st.subheader("📈 Churn by Contract Type")

contract_churn = filtered_df.groupby("Contract")["Churn Value"].mean()

st.bar_chart(contract_churn)

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📋 Data Preview")
st.dataframe(filtered_df.head())
