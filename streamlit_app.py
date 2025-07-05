import streamlit as st
import joblib
import numpy as np

import joblib

model = joblib.load(r"C:\Users\Nanda Chowgle\project\online payment fraud detection\notebook\data\fraud_detection\fraud_model.pkl")
scaler = joblib.load(r"C:\Users\Nanda Chowgle\project\online payment fraud detection\notebook\data\fraud_detection\scaler.pkl")


st.title("💳 Online Payment Fraud Detection")

st.markdown("Enter transaction details below:")

# Input features
step = st.number_input("Step (Hour of Day)", min_value=1, max_value=744, value=1)
type_map = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
transaction_type = st.selectbox("Transaction Type", list(type_map.keys()))
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)
isFlaggedFraud = st.selectbox("Flagged as Fraud by Rule Engine?", [0, 1])

# Prepare input
type_val = type_map[transaction_type]
input_data = np.array([[step, type_val, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFlaggedFraud]])
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {probability:.2%})")