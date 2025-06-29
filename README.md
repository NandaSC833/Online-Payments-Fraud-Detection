# Online Payments Fraud Detection

This project uses machine learning to detect fraudulent online payment transactions. The solution is built with Python, Scikit-learn, and Streamlit to provide a user-friendly web interface for prediction.

---

## Features

- Predicts whether a transaction is **fraudulent or legitimate**
- Trained on a real transaction dataset
- Interactive Streamlit app with custom inputs
- Supports deployment locally or on Streamlit Cloud

---

## Project Structure
Online-Payments-Fraud-Detection/

├── fraud_model.pkl # Trained model

├── scaler.pkl # Feature scaler

├── streamlit_app.py # Streamlit frontend

├── model_training.py # Training script

├── EDA.py # Exploratory Data Analysis

├── requirements.txt # Python dependencies

└── README.md


---

## ML Model

- **Algorithm**: Random Forest Classifier  
- **Features used**:
  - `step`, `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `isFlaggedFraud`

---

## Installation & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py

## Dataset Info
Columns: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Source: Kaggle / Open banking dataset

##Example Inputs

| Feature        | Fraud Example | Legitimate Example |
| -------------- | ------------- | ------------------ |
| type           | TRANSFER      | PAYMENT            |
| amount         | 10000         | 500                |
| oldbalanceOrg  | 10000         | 1000               |
| newbalanceOrig | 0             | 500                |
| oldbalanceDest | 0             | 500                |
| newbalanceDest | 0             | 1000               |


