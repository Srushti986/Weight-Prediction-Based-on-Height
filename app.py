import streamlit as st
import numpy as np
import joblib

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Weight Prediction Based on Height",
    layout="centered"
)

st.title("Height to Weight Prediction")
st.write("Enter your height to predict weight.")

st.divider()

height = st.number_input(
    "Enter Height (inches)",
    min_value=40.0,
    max_value=100.0,
    value=65.0
)

st.progress(int((height - 40) / 60 * 100))
st.caption(f"Selected height: **{height:.1f} inches**")

if st.button("Predict Weight"):
    height_array = np.array([[height]])
    prediction = model.predict(height_array)
    st.success(f"Predicted Weight: {prediction[0]:.2f} pounds")

st.divider()

with st.expander("Model Information"):
    st.write("**Model:** Linear Regression")
    st.write("**Input feature:** Height")
    st.write("**Output feature:** Weight")

st.divider()

with st.expander("How does it work?"):
    st.write("""
    1. Enter your height in inches.
    
    2. The height is given to the trained ML model.
    
    3. Linear Regression calculates the predicted weight.
    
    4. The result is displayed in pounds.
    """)
