import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter the following health metrics to predict your diabetes progression risk.")

with st.form(key="user_info_form"):
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("Body Mass Index (BMI)")
    bp = st.number_input("Average Blood Pressure (BP)")
    tc = st.number_input("Total Cholesterol (TC)")
    ldl = st.number_input("Low Density Lipoproteins (LDL)")
    hdl = st.number_input("High Density Lipoproteins (HDL)")
    tch = st.number_input("Total Cholesterol / HDL (TCH)")
    ltg = st.number_input("Log of Triglycerides (LTG)")
    glu = st.number_input("Blood Glucose Level (GLU)")

    submit_button = st.form_submit_button(label="Predict Your Diabetes Risk")

if submit_button:
    required_fields = [age,gender,bmi, bp, tc, ldl, hdl, tch, ltg, glu]
    if any(val == 0 for val in required_fields):
        st.warning("⚠️ Please make sure all fields are filled with non-zero values.")
    else:
        sex = 1 if gender == "Male" else 2
        # DataFrame with the input
        input_data = pd.DataFrame([[
            age, sex, bmi, bp, tc, ldl, hdl, tch, ltg, glu
        ]], columns=[
            "AGE", "SEX", "BMI", "BP", "S1", "S2", "S3", "S4", "S5","S6"
        ])

        # Predict using model
        prediction = model.predict(input_data)[0]
        st.success(f"🧾 Predicted Diabetes Progression Score: **{prediction:.2f}**")

        st.caption("This score reflects your projected diabetes progression based on medical inputs.")
        if prediction <= 100:
            risk_level = "Low"
            color = "🔵 **Low Risk** (0–100)"
        elif 100 < prediction <= 150:
            risk_level = "Moderate"
            color = "🟢 **Moderate Risk** (101–150)"
        else:
            risk_level = "High"
            color = "🔴 **High Risk** (151+)"

        st.markdown(f"### Risk Category: {color}")
        st.caption("This risk level is derived from the prediction score and helps assess your diabetes progression.")