# Customer Churn Analytics Dashboard

An interactive machine learning dashboard that predicts customer churn and provides actionable insights for retention strategies.

---

## Live App

https://customer-churn-analytics-dashboard.streamlit.app/

---

##  Overview

This project uses an XGBoost model to predict the likelihood of customer churn and presents the results in an interactive Streamlit dashboard.

Users can explore churn behavior dynamically using filters and adjust prediction thresholds in real time.

---

## Key Features

* **Churn Prediction Model** using XGBoost
* **Interactive Dashboard** built with Streamlit
*  **Threshold Tuning** to balance precision and recall
*  **Dynamic Filtering** (Contract Type & Payment Method)
*  **Customer Behavior Insights** with visualizations
*  **High-Risk Customer Identification**

---

##  Dashboard Preview

<img width="1861" height="916" alt="image" src="https://github.com/user-attachments/assets/59ef4f23-a381-48fb-a9b6-c539ca249e49" />


---

##  Model Details

* Algorithm: **XGBoost Classifier**
* Target Variable: `Churn Value`
* Evaluation Focus: Maximizing **recall** for churn detection
* Final Threshold: **0.40**

---

##  Key Insight

Customers with **month-to-month contracts** have the highest churn rate, making them the primary target for retention strategies.

---

##  Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Streamlit

---

##  Project Structure

```
customer-churn-analytics-dashboard/
│
├── app.py
├── xgb_model.pkl
├── Telco_customer_churn.xlsx
├── requirements.txt
├── README.md
```

---

##  Installation

```bash
git clone https://github.com/Olaide07/customer-churn-analytics-dashboard.git
cd customer-churn-analytics-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

##  Business Impact

This dashboard enables stakeholders to:

* Identify customers at high risk of churn
* Understand key drivers of churn
* Make data-driven retention decisions
* Simulate different decision thresholds

---

##  Author

**Olaide Ajibade**

---

##  If you like this project

Give it a star ⭐ and feel free to connect!

