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
        background-color: #f5f7fb;
    }
    .top-bar {
        background: #0f172a;
        color: white;
        padding: 10px 20px;
        border-radius: 0 0 12px 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    .top-bar-left h2 {
        margin: 0;
        font-size: 20px;
    }
    .top-bar-left p {
        margin: 0;
        font-size: 13px;
        opacity: 0.85;
    }
    .top-bar-right {
        font-size: 13px;
        text-align: right;
    }
    .top-bar-right a {
        color: #38bdf8;
        text-decoration: none;
        font-weight: 600;
    }
    .title-container {
        padding: 18px 22px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e88e5, #42a5f5);
        color: white;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.35);
        margin-bottom: 20px;
    }
    .info-card {
        background-color: white;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
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

# ================== TOP DEVELOPER INFO BAR ==================
# ✋ Replace phone number and LinkedIn URL with your real details
st.markdown(
    """
    <div class="top-bar">
        <div class="top-bar-left">
            <h2>Auto MPG – ML Web App</h2>
            <p>End-to-end Machine Learning project deployed with Streamlit</p>
        </div>
        <div class="top-bar-right">
            <div>👨‍💻 <b>Dikesh Chavhan</b></div>
            <div>📞 +91-8591531092</div>
            <div>🔗 <a href="www.linkedin.com/in/dikeshchavhan18" target="_blank">LinkedIn Profile</a></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== HEADER SECTION ==================
st.markdown(
    """
    <div class="title-container">
        <h1>🚗 Car Mileage Prediction</h1>
        <h4>Estimate your car's fuel efficiency (MPG & km/l) using Machine Learning</h4>
        <p>
            This app uses a trained ML model to predict how much mileage a car can give,
            based on its engine and body specifications. Useful for students, car buyers,
            and anyone who wants to understand fuel efficiency.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== SIDEBAR – ABOUT + USER INPUTS ==================
st.sidebar.title("📌 About this App")
st.sidebar.info(
    "This app predicts **car mileage** in MPG (Miles Per Gallon) and also shows "
    "approximate **km/l (kilometres per litre)**.\n\n"
    "It is built on the classic **Auto MPG dataset** and uses a Machine Learning model "
    "trained with Python, scikit-learn and XGBoost."
)

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Enter Car Details")

st.sidebar.write("Fill these details (approximate values are okay):")

# ---- Input widgets ----
cylinders = st.sidebar.selectbox(
    "Number of Cylinders",
    options=[3, 4, 5, 6, 8],
    index=1,
    help="Most Indian passenger cars have 3 or 4 cylinders."
)

displacement = st.sidebar.number_input(
    "Engine Displacement (cc approx.)",
    min_value=50.0, max_value=8000.0, value=1500.0, step=50.0,
    help="Engine size in cc. E.g., 1197, 1498, 1997 etc."
)

horsepower = st.sidebar.number_input(
    "Horsepower (bhp approx.)",
    min_value=30.0, max_value=300.0, value=80.0, step=1.0,
    help="Approximate power of the engine in bhp."
)

weight = st.sidebar.number_input(
    "Vehicle Weight (kg approx.)",
    min_value=600.0, max_value=4000.0, value=1100.0, step=50.0,
    help="Small hatchbacks ~800–1000 kg, SUVs are heavier."
)

acceleration = st.sidebar.number_input(
    "0–60 mph (0–96 kmph) Time (seconds)",
    min_value=5.0, max_value=30.0, value=14.0, step=0.1,
    help="Time taken to reach 60 mph (96 kmph). Lower is faster."
)

model_year = st.sidebar.slider(
    "Model Year (Code: 70–82 from dataset)",
    min_value=70, max_value=82, value=76,
    help="Dataset uses codes 70–82, which roughly map to 1970–1982."
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
    # Column names must match the training phase exactly
    data = {
        "cylinders": [cylinders],
        "displacement": [displacement],
        "horsepower": [horsepower],
        "weight": [weight],
        "acceleration": [acceleration],
        "model year": [model_year],  # assume you trained using 70–82 directly
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
    st.write("Below are the values that will be given to the ML model for prediction:")
    st.dataframe(input_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if model is None:
        st.error(
            "Model file `mpg_prediction_model.pkl` not found.\n\n"
            "👉 Please make sure it is in the same folder as `app.py`."
        )
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            predict_button = st.button("🚀 Predict Mileage")
        with col2:
            st.write("")  # spacer

        if predict_button:
            try:
                prediction = model.predict(input_df)
                mpg = float(prediction[0])
                kmpl = mpg * 0.425144  # simple conversion to km/l

                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.subheader("✅ Predicted Mileage")
                st.markdown(f"### • Estimated Fuel Efficiency: **{mpg:.2f} MPG**")
                st.markdown(f"### • Approximate: **{kmpl:.2f} km/l**")

                # Interpretation for user
                if kmpl > 20:
                    st.info("This is a **highly fuel-efficient** car. Very good for daily city and highway use.")
                elif kmpl > 15:
                    st.info("This car has **average fuel efficiency**, similar to many petrol cars in India.")
                else:
                    st.info("This car seems **less fuel-efficient**. Fuel expenses may be higher over time.")

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error("⚠️ Prediction failed. Please check that the model and input features match.")
                st.exception(e)

# -------- TAB 2: HOW THIS APP WORKS --------
with tab2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ What does this app do?")

    st.write(
        """
        - This app predicts **car mileage** using a trained **Machine Learning model**.  
        - It takes common car specifications as input and outputs the expected **MPG** and equivalent **km/l**.  
        - The model is trained on the classic **Auto MPG dataset** from the UCI Machine Learning Repository.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🧠 How does it work internally?")

    st.write(
        """
        1. Data cleaning (handling missing values like horsepower).  
        2. Outlier removal using statistical methods (IQR).  
        3. Feature engineering – scaling numeric features and encoding categorical variables.  
        4. Multiple algorithms tested: **Linear Regression, Lasso, Ridge, XGBoost**.  
        5. Best model (XGBoost + preprocessing pipeline) saved as `mpg_prediction_model.pkl`.  
        6. This Streamlit app loads the saved model and makes predictions in real-time.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🎓 Who is this useful for?")

    st.write(
        """
        - **Students**: End-to-end ML project (from dataset to deployment).  
        - **Car buyers**: Rough idea of mileage based on specs.  
        - **Data science portfolio**: Can be linked in GitHub, LinkedIn, and resume.
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
