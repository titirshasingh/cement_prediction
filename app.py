
import streamlit as st
import numpy as np
import joblib

# Load your trained model and scaler
model = joblib.load("catboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page setup
st.set_page_config(page_title="Concrete Strength Predictor", layout="centered")
st.title("🧱 Concrete Strength Prediction App")
st.markdown("Enter the concrete mix parameters to predict its compressive strength (in MPa).")

# Input fields
cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=300.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0, value=0.0)
fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0)
water = st.number_input("Water (kg/m³)", min_value=0.0, value=180.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, value=10.0)
coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0, value=950.0)
fine_agg = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0, value=800.0)
age = st.number_input("Age of concrete (days)", min_value=1, value=28)

# Predict button
if st.button("Predict Strength"):
    input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    st.success(f"Predicted Compressive Strength: **{prediction[0]:.2f} MPa**")
