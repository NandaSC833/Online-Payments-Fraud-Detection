import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")  # Save your scaler during training

st.title("💳 Online Payment Fraud Detection")

st.write("Enter transaction details below to predict if it's fraudulent.")

# User inputs
type_map = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}

type_input = st.selectbox("Transaction Type", list(type_map.keys()))
amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0)

# Prepare features
if st.button("Predict Fraud"):
    type_val = type_map[type_input]
    features = np.array([[type_val, amount, oldbalanceOrg, newbalanceOrig,
                          oldbalanceDest, newbalanceDest]])

    # Scale inputs
    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]
    prob = model.predict_proba(scaled_features)[0][1]

    if prediction == 1:
        st.error(f"Fraud Detected! Probability: {prob:.2%}")
    else:
        st.success(f"Transaction Looks Safe. Probability of Fraud: {prob:.2%}")
        