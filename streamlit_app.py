import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load(r"C:\Users\Nanda Chowgle\project\online payment fraud detection\fraud_model.pkl")
scaler = joblib.load(r"C:\Users\Nanda Chowgle\project\online payment fraud detection\scaler.pkl")

st.title("💳 Online Payment Fraud Detection")
st.write("Enter the transaction details below to check if it's potentially fraudulent.")

# Transaction type mapping
type_map = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}

# Input fields
step = st.number_input("Step (time index)", min_value=1, value=1)
type_str = st.selectbox("Transaction Type", list(type_map.keys()))
type_code = type_map[type_str]

amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0)
isFlaggedFraud = st.selectbox("Is Flagged Fraud (0 or 1)", [0, 1])

# Predict button
if st.button("Predict Fraud"):
    try:
        features = np.array([[step, type_code, amount, oldbalanceOrg, newbalanceOrig,
                              oldbalanceDest, newbalanceDest, isFlaggedFraud]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        if prediction == 1:
            st.error(f"🚨 Fraudulent Transaction Detected! (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Transaction Looks Safe (Probability of Fraud: {probability:.2f})")
    except Exception as e:
        st.warning(f"⚠️ Prediction failed: {e}")
        