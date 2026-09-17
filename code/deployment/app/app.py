import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000")

st.title("Telco Customer Churn")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col2:
    online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
    paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
    payment_method = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total charges ($)", min_value=0.0, value=840.0)

if st.button("Predict churn"):
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    response = requests.post(f"{API_URL}/predict", json=payload)
    response.raise_for_status()
    result = response.json()

    st.metric("Churn probability", f"{result['churn_probability']:.1%}")
    if result["churn"]:
        st.error("High churn risk")
    else:
        st.success("Low churn risk")
