import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title=" House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ==============================
# Load Model & Scaler
# ==============================
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# Title
# ==============================
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>House Price Prediction App</h1>
<p style='text-align: center; font-size:18px;'>
Enter property details to estimate the market price.
</p>
""", unsafe_allow_html=True)

st.divider()

# ==============================
# Input Layout (Side by Side)
# ==============================
st.subheader("🏡 Enter Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    square_footage = st.number_input("Square Footage", 300, 10000, 2000)
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)

with col2:
    year_built = st.number_input("Year Built", 1900, 2025, 2010)
    lot_size = st.number_input("Lot Size", 0.1, 10.0, 2.0)
    garage_size = st.number_input("Garage Size", 0, 5, 1)

with col3:
    neighborhood_quality = st.slider("Neighborhood Quality", 1, 10, 7)

st.divider()

# ==============================
# Prediction
# ==============================
if st.button("🔍 Predict House Price", use_container_width=True):

    input_data = pd.DataFrame([{
        "Square_Footage": square_footage,
        "Num_Bedrooms": bedrooms,
        "Num_Bathrooms": bathrooms,
        "Year_Built": year_built,
        "Lot_Size": lot_size,
        "Garage_Size": garage_size,
        "Neighborhood_Quality": neighborhood_quality
    }])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.markdown("## 💰 Estimated House Price")
    st.success(f"### ${prediction:,.2f}")

st.divider()
st.caption("Built with Streamlit | Model: Linear Regression")