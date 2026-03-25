import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# =========================

# PAGE CONFIG

# =========================

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.title("Customer Churn Analytics Dashboard")
st.markdown("### Built by Olaide Ajibade")

# =========================

# SIDEBAR

# =========================

st.sidebar.title("Controls")

threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.4)

# =========================

# LOAD MODEL

# =========================

@st.cache_resource
def load_model():
    with open("xgb_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# =========================

# LOAD DATA

# =========================

@st.cache_data
def load_data():
    return pd.read_excel("Telco_customer_churn.xlsx")

df = load_data()

# =========================

# PREPROCESS

# =========================

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())

cols_to_drop = [
"CustomerID","Count","Country","State","City","Zip Code",
"Lat Long","Latitude","Longitude","Churn Label",
"Churn Score","CLTV","Churn Reason"
]
df = df.drop(columns=cols_to_drop)

# =========================

# FEATURES

# =========================

X = df.drop(columns=["Churn Value"])
X = pd.get_dummies(X, drop_first=True)

for col in model.feature_names_in_:
    if col not in X.columns:
        X[col] = 0

X = X[model.feature_names_in_]

# =========================

# PREDICTIONS

# =========================


y_prob = model.predict_proba(X.values)[:, 1]
df["Churn_Probability"] = y_prob
df["Predicted_Churn"] = (df["Churn_Probability"] > threshold).astype(int)

# =========================

# FILTERS

# =========================

st.sidebar.subheader("Filters")

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

filtered_df = df[
(df["Contract"].isin(contract_filter)) &
(df["Payment Method"].isin(payment_filter))
]

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# =========================

# METRICS

# =========================

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(filtered_df))
col2.metric("Actual Churn Rate", f"{filtered_df['Churn Value'].mean()*100:.2f}%")
col3.metric(
"Predicted Churn",
int(filtered_df["Predicted_Churn"].sum()),
f"{filtered_df['Predicted_Churn'].mean()*100:.1f}%"
)

# =========================

# INSIGHTS HEADER

# =========================

st.markdown("---")
st.header("Customer Behavior Insights")

# =========================

# CHARTS (SIDE BY SIDE)

# =========================

colA, colB = st.columns(2)

# ---- Contract Chart ----

with colA:
    st.subheader("Churn by Contract")


contract_churn = (
    filtered_df.groupby("Contract")["Churn Value"]
    .mean()
    .sort_values(ascending=False)
)

fig1, ax1 = plt.subplots(figsize=(5,4))
contract_churn.plot(kind="bar", ax=ax1)
ax1.set_ylabel("Churn Rate")

for i, v in enumerate(contract_churn):
    ax1.text(i, v, f"{v:.2f}", ha='center')

    st.pyplot(fig1)


# ---- Payment Chart ----

with colB:
    st.subheader("Churn by Payment Method")


payment_churn = (
    filtered_df.groupby("Payment Method")["Churn Value"]
    .mean()
    .sort_values(ascending=False)
)

fig2, ax2 = plt.subplots(figsize=(5,4))
payment_churn.plot(kind="bar", ax=ax2)
ax2.set_ylabel("Churn Rate")

st.pyplot(fig2)


# =========================

# MONTHLY CHARGES

# =========================

st.markdown("---")
st.subheader("Monthly Charges Distribution")

fig3, ax3 = plt.subplots(figsize=(6,4))

filtered_df[filtered_df["Churn Value"] == 1]["Monthly Charges"].hist(
alpha=0.5, label="Churn", ax=ax3
)
filtered_df[filtered_df["Churn Value"] == 0]["Monthly Charges"].hist(
alpha=0.5, label="No Churn", ax=ax3
)

ax3.legend()
st.pyplot(fig3)

# =========================

# KEY INSIGHT

# =========================

st.markdown("---")
st.subheader("Key Insight")

top_contract = contract_churn.idxmax()
top_rate = contract_churn.max()

st.success(
f"Highest churn occurs in **{top_contract} contracts** at {top_rate:.2%}. Focus retention strategies here."
)

# =========================

# HIGH RISK CUSTOMERS

# =========================

st.markdown("---")
st.subheader("High Risk Customers")

high_risk = filtered_df[filtered_df["Churn_Probability"] > 0.7]

st.dataframe(
high_risk[
["Contract", "Payment Method", "Monthly Charges", "Churn_Probability"]
]
.sort_values(by="Churn_Probability", ascending=False)
.head(20)
)
