import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/heart_model.pkl')
columns = joblib.load('models/feature_columns.pkl')
encoders = joblib.load('models/encoders.pkl')

st.title("Heart Disease Risk Predictor")
st.write("Enter patient details to predict heart disease risk.")

age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex", encoders['sex'].classes_)
dataset_source = st.selectbox("Source Hospital", encoders['dataset'].classes_)
cp = st.selectbox("Chest Pain Type", encoders['cp'].classes_)
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", encoders['fbs'].classes_)
restecg = st.selectbox("Resting ECG", encoders['restecg'].classes_)
thalch = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise-Induced Angina", encoders['exang'].classes_)
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0)

if st.button("Predict"):
    input_dict = {
        'age': age,
        'sex': encoders['sex'].transform([sex])[0],
        'dataset': encoders['dataset'].transform([dataset_source])[0],
        'cp': encoders['cp'].transform([cp])[0],
        'trestbps': trestbps,
        'chol': chol,
        'fbs': encoders['fbs'].transform([fbs])[0],
        'restecg': encoders['restecg'].transform([restecg])[0],
        'thalch': thalch,
        'exang': encoders['exang'].transform([exang])[0],
        'oldpeak': oldpeak,
    }
    input_df = pd.DataFrame([input_dict])[columns]  # reorder to match training columns exactly
    pred = model.predict(input_df)[0]
    result = "⚠️ High risk of heart disease" if pred == 1 else "✅ Low risk of heart disease"
    st.subheader(result)