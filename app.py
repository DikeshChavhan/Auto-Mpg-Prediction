import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Car Mileage (MPG) Prediction",
    page_icon="🚗",
    layout="centered"
)

# ================== CUSTOM CSS FOR BEAUTIFUL UI ==================
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f7f7;
    }
    .title-container {
        padding: 20px 25px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e88e5, #42a5f5);
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 20px;
    }
    .info-card {
        background-color: white;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
    .result-box {
        background: #e3f2fd;
        border-left: 6px solid #1e88e5;
        padding: 15px 20px;
        border-radius: 10px;
        margin-top: 15px;
    }
    .footer {
        font-size: 13px;
        color: #666666;
        margin-top: 30px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== LOAD TRAINED MODEL ==================
@st.cache_resource
def load_model():
    try:
        with open("mpg_prediction_model.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

model = load_model()

# ================== HEADER SECTION ==================
st.markdown(
    """
    <div class="title-container">
        <h1>🚗 Car Mileage Prediction App</h1>
        <h4>Estimate your car's fuel efficiency (MPG) using Machine Learning</h4>
        <p>Enter the car details on the left and get an instant mileage prediction.
        This can help you compare cars, understand fuel costs, and make smarter decisions.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== SIDEBAR – USER INPUTS ==================
st.sidebar.title("⚙️ Enter Car Details")

st.sidebar.write("Fill these details based on your car or the car you are planning to buy:")

cylinders = st.sidebar.selectbox(
    "Number of Cylinders",
    options=[3, 4, 5, 6, 8],
    index=1,
    help="Most Indian cars have 3 or 4 cylinders."
)

displacement = st.sidebar.number_input(
    "Engine Displacement (cc approx.)",
    min_value=50.0, max_value=8000.0, value=1500.0, step=50.0,
    help="Engine size in cubic centimetres (cc). Example: 1197, 1498, 1997, etc."
)

horsepower = st.sidebar.number_input(
    "Horsepower (bhp approx.)",
    min_value=30.0, max_value=300.0, value=80.0, step=1.0,
    help="Approximate power of the engine in bhp."
)

weight = st.sidebar.number_input(
    "Vehicle Weight (kg approx.)",
    min_value=600.0, max_value=4000.0, value=1100.0, step=50.0,
    help="Approximate weight of the car. Small hatchbacks are ~800–1000 kg."
)

acceleration = st.sidebar.number_input(
    "0–60 mph (0–96 kmph) Time (seconds)",
    min_value=5.0, max_value=30.0, value=14.0, step=0.1,
    help="How many seconds it takes to reach 60 mph (96 kmph)."
)

model_year = st.sidebar.slider(
    "Model Year (Approximate)",
    min_value=1970, max_value=1982, value=1978,
    help="Dataset is from older years, but you can choose nearest year."
)

origin_display = st.sidebar.selectbox(
    "Car Origin / Region",
    ["USA (1)", "Europe (2)", "Japan / Asia (3)"],
    help="Choose the region where the car is manufactured."
)

origin_map = {
    "USA (1)": 1,
    "Europe (2)": 2,
    "Japan / Asia (3)": 3
}
origin = origin_map[origin_display]

# ================== INPUT DATAFRAME FOR MODEL ==================
def make_input_df():
    # NOTE: Column names must exactly match the names used during training
    data = {
        "cylinders": [cylinders],
        "displacement": [displacement],
        "horsepower": [horsepower],
        "weight": [weight],
        "acceleration": [acceleration],
        "model year": [model_year - 1900],  # if you trained with 70–82 as given in dataset, adjust here
        # If your model was trained directly on 70–82 without +1900, comment out above and use below line instead:
        # "model year": [model_year],
        "origin": [origin],
    }
    return pd.DataFrame(data)

input_df = make_input_df()

# ================== MAIN CONTENT – TABS ==================
tab1, tab2 = st.tabs(["📊 Prediction", "ℹ️ How this app works"])

# -------- TAB 1: PREDICTION --------
with tab1:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🔎 Car Details Summary")
    st.write("These are the details that will be used by the model to estimate mileage:")
    st.dataframe(input_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if model is None:
        st.error(
            "Model file `mpg_prediction_model.pkl` not found. "
            "Please make sure it is in the same folder as `app.py`."
        )
    else:
        predict_button = st.button("🚀 Predict Mileage (MPG)")

        if predict_button:
            try:
                prediction = model.predict(input_df)
                mpg = float(prediction[0])

                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("✅ Predicted Mileage")
                st.markdown(f"### Estimated Fuel Efficiency: **{mpg:.2f} MPG**")

                # Optional helpful explanation (rough conversion)
                kmpl = mpg * 0.425144  # simple conversion to km/l
                st.write(f"That is approximately **{kmpl:.2f} km/l** (kilometres per litre).")

                # Interpretation for user
                if kmpl > 20:
                    st.info("This is a **highly fuel-efficient** car. Good for daily city and highway use.")
                elif kmpl > 15:
                    st.info("This car has **average fuel efficiency**, similar to many petrol cars in India.")
                else:
                    st.info("This car seems **less fuel-efficient**. Fuel costs may be higher over time.")

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error("⚠️ Prediction failed. Please check that the model and input features match.")
                st.exception(e)

# -------- TAB 2: HOW THIS APP WORKS --------
with tab2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ About this App")

    st.write(
        """
        **What does this app do?**  
        This app uses a Machine Learning model trained on the famous **Auto MPG dataset**
        to estimate the **mileage (MPG – Miles Per Gallon)** of a car using its specifications.

        **Who can use this app?**
        - Students learning Machine Learning & Data Science  
        - Car enthusiasts who want to understand how specs affect mileage  
        - Anyone curious about fuel efficiency and car performance  

        **What inputs are required?**
        - Number of cylinders  
        - Engine size (displacement)  
        - Horsepower (bhp)  
        - Vehicle weight  
        - Acceleration time  
        - Model year  
        - Region / origin of the car  

        The model then predicts:
        - **MPG (Miles Per Gallon)** – a standard mileage measure  
        - An approximate conversion to **km/l (kilometres per litre)** for easier understanding in India.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🧠 Behind the Scenes")

    st.write(
        """
        - Data cleaned, missing values handled  
        - Outliers treated using statistical methods (IQR)  
        - Features scaled and encoded using **scikit-learn Pipelines**  
        - Multiple models tested: Linear Regression, Lasso, Ridge, XGBoost  
        - Best performing model saved as `mpg_prediction_model.pkl` and used in this app  

        This demonstrates a complete **end-to-end ML project**:
        from data to deployment.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown(
    """
    <div class="footer">
        Made with ❤️ in India using Streamlit & Machine Learning<br>
        Developer: <b>Dikesh Chavhan</b>
    </div>
    """,
    unsafe_allow_html=True
)
