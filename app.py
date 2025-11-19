import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ================== Load Trained Model ==================
@st.cache_resource
def load_model():
    with open("mpg_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ================== App UI ==================
st.set_page_config(page_title="Car MPG Prediction App", page_icon="🚗", layout="centered")

st.title("🚗 Car MPG Prediction App")
st.write(
    """
This app predicts the **Miles Per Gallon (MPG)** of a car  
based on its specifications using a trained Machine Learning model.
"""
)

st.sidebar.header("Input Features")

# Sidebar inputs
cylinders = st.sidebar.selectbox("Cylinders", [3, 4, 5, 6, 8])
displacement = st.sidebar.number_input("Displacement (cu-in)", min_value=50.0, max_value=500.0, value=150.0, step=1.0)
horsepower = st.sidebar.number_input("Horsepower", min_value=40.0, max_value=250.0, value=90.0, step=1.0)
weight = st.sidebar.number_input("Weight (lbs)", min_value=1500.0, max_value=5500.0, value=2500.0, step=50.0)
acceleration = st.sidebar.number_input("Acceleration (0–60 mph time)", min_value=8.0, max_value=30.0, value=15.0, step=0.1)
model_year = st.sidebar.slider("Model Year", min_value=70, max_value=82, value=76)

origin_display = st.sidebar.selectbox(
    "Origin",
    ["USA (1)", "Europe (2)", "Japan (3)"]
)

# Map origin text to numeric code
origin_map = {
    "USA (1)": 1,
    "Europe (2)": 2,
    "Japan (3)": 3
}
origin = origin_map[origin_display]

# ================== Build Input DataFrame ==================
def make_input_df():
    data = {
        "cylinders": [cylinders],
        "displacement": [displacement],
        "horsepower": [horsepower],
        "weight": [weight],
        "acceleration": [acceleration],
        "model year": [model_year],
        "origin": [origin],
    }
    return pd.DataFrame(data)

input_df = make_input_df()

st.subheader("🔎 Input Features Used for Prediction")
st.dataframe(input_df)

# ================== Prediction ==================
if st.button("Predict MPG"):
    try:
        prediction = model.predict(input_df)
        mpg = prediction[0]

        st.subheader("✅ Predicted MPG")
        st.success(f"Estimated fuel efficiency: **{mpg:.2f} MPG**")

        # Simple interpretation
        if mpg > 30:
            st.info("This is a **highly fuel-efficient** car.")
        elif mpg > 20:
            st.info("This car has **average fuel efficiency**.")
        else:
            st.info("This car is **less fuel-efficient**.")

    except Exception as e:
        st.error("Prediction failed. Please check that the model and input columns match.")
        st.exception(e)

st.markdown("---")
st.caption("Developed by **Dikesh Chavhan** – Auto MPG ML Project")
