import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

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
    return pd.read_excel("Telco_customer_churn.xlsx")

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

    # Encode
    df = pd.get_dummies(df)

    return df

df_encoded = preprocess(df)

# -------------------------------
# ALIGN FEATURES
# -------------------------------
model_features = model.get_booster().feature_names

for col in model_features:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

df_encoded = df_encoded[model_features]

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("⚙️ Controls")

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.4)

st.sidebar.markdown("### 🔎 Filters")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    df["Contract"].unique(),
    default=df["Contract"].unique()
)

payment_filter = st.sidebar.multiselect(
    "Payment Method",
    df["Payment Method"].unique(),
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
# PREDICTION (SAFE)
# -------------------------------
dmatrix = xgb.DMatrix(df_encoded, feature_names=model_features)

y_prob = model.get_booster().predict(dmatrix)
y_pred = (y_prob >= threshold).astype(int)

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 Customer Churn Analytics Dashboard")

st.markdown(
"""
**Built by Olaide Ajibade**  
Machine Learning | Data Science | Predictive Analytics  

This dashboard predicts customer churn and provides insights into key drivers such as contract type and payment method.
"""
)

# -------------------------------
# METRICS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Actual Churn Rate", f"{df['Churn Value'].mean()*100:.2f}%")
col3.metric("Predicted Churn", int(y_pred.sum()))

st.markdown("---")

# -------------------------------
# CHART
# -------------------------------
st.subheader("📈 Churn by Contract Type")

contract_churn = filtered_df.groupby("Contract")["Churn Value"].mean()
st.bar_chart(contract_churn)

# -------------------------------
# INSIGHT
# -------------------------------
st.subheader("💡 Key Insight")

month_churn = df[df["Contract"] == "Month-to-month"]["Churn Value"].mean()

st.info(
    f"Customers on month-to-month contracts have a churn rate of {month_churn*100:.1f}%, "
    f"significantly higher than long-term contracts."
)

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📋 Data Preview")
st.dataframe(filtered_df.head())
