import streamlit as st
import pickle
import numpy as np
import os

# 1. Page Configuration (App ki look aur styling ke liye)
st.set_page_config(
    page_title="Starbucks Price Predictor",
    page_icon="☕",
    layout="centered"
)

# App ka Title aur Introduction
st.title("☕ Starbucks Price Predictor App")
st.write("Decision Tree Regressor Model ka use karke continuous values ko predict karein.")
st.divider()

# 2. Model Loading Logic with Error Handling
model_file = 'starbucks_model.pkl'
model = None

if not os.path.exists(model_file):
    st.error(f"❌ **Error:** '{model_file}' file aapki GitHub repository me nahi mili! Kripya check karein ki file ka naam bilkul sahi ho.")
else:
    try:
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        st.success("✅ Machine Learning Model successfully load ho gaya hai!")
    except Exception as e:
        st.error(f"❌ **Technical Error:** Model load karne me masla aaya: {e}")

st.divider()

# 3. User Input Fields (Aapke Dataset aur Model ke Features)
st.subheader("📝 Predictions Ke Liye Details Enter Karein:")

# Quantity input (Default value: 5.0)
quantity = st.number_input(
    "Quantity:", 
    min_value=1.0, 
    max_value=100.0, 
    value=5.0, 
    step=1.0,
    help="Product ki quantity enter karein."
)

# Base Price input (Default value: 5.0)
price = st.number_input(
    "Base Price ($):", 
    min_value=0.1, 
    max_value=1000.0, 
    value=5.0, 
    step=0.5,
    help="Product ki base price dollar ($) me enter karein."
)

st.divider()

# 4. Prediction Button aur Logic
if st.button("Predict Target Value", type="primary"):
    if model is not None:
        try:
            # Inputs ko 2D array me convert karna (Decision tree ki requirement)
            features = np.array([[quantity, price]])
            
            # Model se prediction lena
            prediction = model.predict(features)
            
            # Output screen par show karna
            st.balloons() # Ek choti si animation successfully run hone par
            st.success(f"🎯 **Predicted Output Value:** `${prediction[0]:.2f}`")
            
       
