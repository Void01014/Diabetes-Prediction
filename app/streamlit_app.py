import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

st.title("Diabetes Risk Prediction")
st.caption("Enter the patient's clinical values")

col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=6, step=1)    
    Glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=140.0)
    BloodPressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=72.0)
    SkinThickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=35.0)
with col2:
    Insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=218.0)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=33.6)
    DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.627)
    Age = st.number_input("Age", min_value=1, max_value=120, value=50, step=1)

if st.button("Predict"):
    payload = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age,
    }

    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()

        if result["risk_category"] == 1:
            st.error("High risk of diabetes.")
            st.write("Follow-up: repeat a fasting glucose test, book a consultation with a doctor, and review diet and physical activity.")
        else:
            st.success("Low risk of diabetes.")
            st.write("Follow-up: no action needed now. Keep a healthy lifestyle and do a routine check-up each year.")
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the API. Please ensure the FastAPI server is running.")