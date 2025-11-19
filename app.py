import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Car Mileage (MPG) Prediction",
    page_icon="🚗",
    layout="wide"
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

# ================== SESSION STATE FOR HISTORY ==================
if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []  # list of dicts

# ================== TOP DEVELOPER INFO BAR ==================
# 👉 Replace phone number and LinkedIn URL with your real details
# ================== TOP DEVELOPER INFO BAR ==================
st.markdown(
    """
    <div class="top-bar">
        <div class="top-bar-left">
            <h2>Auto MPG – ML Web App</h2>
            <p>End-to-end Machine Learning project deployed using Streamlit</p>
        </div>
        <div class="top-bar-right">
            <div>👨‍💻 <b>Dikesh Chavhan</b></div>
            <div>📞 <a href="tel:+918591531092" style="color:#38bdf8;text-decoration:none;font-weight:600;">+91-8591531092</a></div>
            <div>🔗 <a href="https://www.linkedin.com/in/dikeshchavhan18" target="_blank" style="color:#38bdf8;text-decoration:none;font-weight:600;">LinkedIn Profile</a></div>
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
        <h4>Estimate your car's fuel efficiency (MPG & km/l) and fuel cost using Machine Learning</h4>
        <p>
            This app predicts how much mileage a car can give based on its engine and body specifications.
            It is designed for Indian users, students and car buyers to understand fuel efficiency and running cost.
        </p>
        <p><b>नोट:</b> ज़्यादा <b>km/l</b> मतलब गाड़ी पेट्रोल कम खाएगी और आपका खर्चा कम होगा।</p>
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
    "trained with Python, scikit-learn and XGBoost.\n\n"
    "फ्यूल एफिशिएंसी समझने और कार compare करने के लिए ये app useful है।"
)

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Car A – Enter Details")

# ---- Car A Input widgets (sidebar) ----
cylinders_A = st.sidebar.selectbox(
    "Number of Cylinders",
    options=[3, 4, 5, 6, 8],
    index=1,
    help="Most Indian passenger cars have 3 or 4 cylinders."
)

displacement_A = st.sidebar.number_input(
    "Engine Displacement (cc approx.)",
    min_value=50.0, max_value=8000.0, value=1500.0, step=50.0,
    help="Engine size in cc. E.g., 1197, 1498, 1997 etc."
)

horsepower_A = st.sidebar.number_input(
    "Horsepower (bhp approx.)",
    min_value=30.0, max_value=300.0, value=80.0, step=1.0,
    help="Approximate power of the engine in bhp."
)

weight_A = st.sidebar.number_input(
    "Vehicle Weight (kg approx.)",
    min_value=600.0, max_value=4000.0, value=1100.0, step=50.0,
    help="Small hatchbacks ~800–1000 kg, SUVs are heavier."
)

acceleration_A = st.sidebar.number_input(
    "0–60 mph (0–96 kmph) Time (seconds)",
    min_value=5.0, max_value=30.0, value=14.0, step=0.1,
    help="Time taken to reach 60 mph (96 kmph). Lower is faster."
)

model_year_A = st.sidebar.slider(
    "Model Year (Code: 70–82 from dataset)",
    min_value=70, max_value=82, value=76,
    help="Dataset uses codes 70–82, which roughly map to 1970–1982."
)

origin_display_A = st.sidebar.selectbox(
    "Car Origin / Region",
    ["USA (1)", "Europe (2)", "Japan / Asia (3)"],
    help="Choose the region where the car is manufactured."
)

origin_map = {
    "USA (1)": 1,
    "Europe (2)": 2,
    "Japan / Asia (3)": 3
}
origin_A = origin_map[origin_display_A]

# ---- Fuel cost settings ----
st.sidebar.markdown("---")
st.sidebar.title("⛽ Fuel Cost Settings")

monthly_km = st.sidebar.number_input(
    "Average Monthly Running (km)",
    min_value=200, max_value=10000, value=1000, step=100,
    help="How many kilometres you drive per month."
)

fuel_price = st.sidebar.number_input(
    "Fuel Price (₹ per litre)",
    min_value=50.0, max_value=200.0, value=105.0, step=1.0,
    help="Current petrol/diesel price per litre in your city."
)

# ================== FUNCTIONS ==================
def make_input_df(cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
    """Build a single-row DataFrame with correct column names."""
    return pd.DataFrame({
        "cylinders": [cylinders],
        "displacement": [displacement],
        "horsepower": [horsepower],
        "weight": [weight],
        "acceleration": [acceleration],
        "model year": [model_year],  # trained using 70–82 directly
        "origin": [origin],
    })

def compute_fuel_cost(kmpl, monthly_km, fuel_price):
    """Return monthly and yearly fuel cost given kmpl."""
    if kmpl <= 0:
        return None, None
    litres_per_month = monthly_km / kmpl
    monthly_cost = litres_per_month * fuel_price
    yearly_cost = monthly_cost * 12
    return monthly_cost, yearly_cost

# ================== MAIN CONTENT – TABS ==================
tab1, tab2, tab3 = st.tabs(["📊 Prediction & Comparison", "📈 Insights & Feature Importance", "ℹ️ How this app works"])

# -------- TAB 1: PREDICTION & COMPARISON --------
with tab1:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🧩 Choose Mode")
    mode = st.radio(
        "Select what you want to do:",
        ["Single Car Prediction", "Compare Two Cars (A vs B)"],
        horizontal=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ========== Single Car Mode ==========
    if mode == "Single Car Prediction":
        input_df_A = make_input_df(
            cylinders_A, displacement_A, horsepower_A, weight_A,
            acceleration_A, model_year_A, origin_A
        )

        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("🔎 Car A Details")
        st.write("These values will be used by the ML model for prediction:")
        st.dataframe(input_df_A, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if model is None:
            st.error(
                "Model file `mpg_prediction_model.pkl` not found.\n\n"
                "👉 Please make sure it is in the same folder as `app.py`."
            )
        else:
            if st.button("🚀 Predict Mileage for Car A"):
                try:
                    prediction = model.predict(input_df_A)
                    mpg = float(prediction[0])
                    kmpl = mpg * 0.425144  # conversion

                    monthly_cost, yearly_cost = compute_fuel_cost(kmpl, monthly_km, fuel_price)

                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("✅ Car A – Predicted Mileage & Fuel Cost")
                    st.markdown(f"### • Estimated Fuel Efficiency: **{mpg:.2f} MPG**")
                    st.markdown(f"### • Approximate: **{kmpl:.2f} km/l**")

                    if monthly_cost is not None:
                        st.markdown(
                            f"**Estimated Fuel Cost:**\n\n"
                            f"- Monthly: **₹{monthly_cost:,.0f}**\n"
                            f"- Yearly: **₹{yearly_cost:,.0f}**"
                        )

                    st.write("ज़्यादा km/l ⇒ कम पेट्रोल लगेगा ⇒ आपका monthly खर्चा कम होगा।")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Save to history
                    st.session_state["prediction_history"].append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Mode": "Single",
                        "Car": "A",
                        "MPG": round(mpg, 2),
                        "KMPL": round(kmpl, 2),
                        "Monthly_km": monthly_km,
                        "Fuel_price": fuel_price,
                        "Monthly_cost": round(monthly_cost, 2) if monthly_cost else None
                    })

                except Exception as e:
                    st.error("⚠️ Prediction failed. Please check that the model and input features match.")
                    st.exception(e)

    # ========== Compare Two Cars Mode ==========
    else:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("🚗 Compare Car A vs Car B")

        colA, colB = st.columns(2)

        # Car A summary (from sidebar values)
        with colA:
            st.markdown("### Car A")
            input_df_A = make_input_df(
                cylinders_A, displacement_A, horsepower_A, weight_A,
                acceleration_A, model_year_A, origin_A
            )
            st.dataframe(input_df_A, use_container_width=True)

        # Car B input controls
        with colB:
            st.markdown("### Car B – Enter Details")
            cylinders_B = st.selectbox("Cylinders (B)", [3, 4, 5, 6, 8], index=1, key="cyl_B")
            displacement_B = st.number_input("Displacement cc (B)", 50.0, 8000.0, 1200.0, 50.0, key="disp_B")
            horsepower_B = st.number_input("Horsepower bhp (B)", 30.0, 300.0, 75.0, 1.0, key="hp_B")
            weight_B = st.number_input("Weight kg (B)", 600.0, 4000.0, 1000.0, 50.0, key="wt_B")
            acceleration_B = st.number_input("0–60 mph (B)", 5.0, 30.0, 15.0, 0.1, key="acc_B")
            model_year_B = st.slider("Model Year code (B)", 70, 82, 78, key="my_B")
            origin_display_B = st.selectbox("Origin (B)", ["USA (1)", "Europe (2)", "Japan / Asia (3)"], key="org_B")
            origin_B = origin_map[origin_display_B]

        input_df_B = make_input_df(
            cylinders_B, displacement_B, horsepower_B, weight_B,
            acceleration_B, model_year_B, origin_B
        )

        st.markdown('</div>', unsafe_allow_html=True)

        if model is None:
            st.error(
                "Model file `mpg_prediction_model.pkl` not found.\n\n"
                "👉 Please make sure it is in the same folder as `app.py`."
            )
        else:
            if st.button("⚖️ Compare Car A and Car B"):
                try:
                    pred_A = model.predict(input_df_A)[0]
                    pred_B = model.predict(input_df_B)[0]

                    mpg_A, mpg_B = float(pred_A), float(pred_B)
                    kmpl_A, kmpl_B = mpg_A * 0.425144, mpg_B * 0.425144

                    cost_A_month, cost_A_year = compute_fuel_cost(kmpl_A, monthly_km, fuel_price)
                    cost_B_month, cost_B_year = compute_fuel_cost(kmpl_B, monthly_km, fuel_price)

                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("📊 Comparison Result – Mileage & Fuel Cost")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("### 🚗 Car A")
                        st.write(f"**Mileage:** {mpg_A:.2f} MPG ({kmpl_A:.2f} km/l)")
                        st.write(f"**Monthly Fuel Cost:** ₹{cost_A_month:,.0f}")
                        st.write(f"**Yearly Fuel Cost:** ₹{cost_A_year:,.0f}")

                    with c2:
                        st.markdown("### 🚙 Car B")
                        st.write(f"**Mileage:** {mpg_B:.2f} MPG ({kmpl_B:.2f} km/l)")
                        st.write(f"**Monthly Fuel Cost:** ₹{cost_B_month:,.0f}")
                        st.write(f"**Yearly Fuel Cost:** ₹{cost_B_year:,.0f}")

                    if kmpl_A > kmpl_B:
                        st.success("✅ **Car A** is more fuel-efficient and will cost you less in fuel.")
                    elif kmpl_B > kmpl_A:
                        st.success("✅ **Car B** is more fuel-efficient and will cost you less in fuel.")
                    else:
                        st.info("Both cars have almost the **same fuel efficiency**.")

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Save to history (A and B)
                    st.session_state["prediction_history"].append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Mode": "Compare",
                        "Car": "A",
                        "MPG": round(mpg_A, 2),
                        "KMPL": round(kmpl_A, 2),
                        "Monthly_km": monthly_km,
                        "Fuel_price": fuel_price,
                        "Monthly_cost": round(cost_A_month, 2)
                    })
                    st.session_state["prediction_history"].append({
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Mode": "Compare",
                        "Car": "B",
                        "MPG": round(mpg_B, 2),
                        "KMPL": round(kmpl_B, 2),
                        "Monthly_km": monthly_km,
                        "Fuel_price": fuel_price,
                        "Monthly_cost": round(cost_B_month, 2)
                    })

                except Exception as e:
                    st.error("⚠️ Comparison failed. Please check model and inputs.")
                    st.exception(e)

    # ========== Show prediction history ==========
    if st.session_state["prediction_history"]:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📜 Recent Predictions (this session)")
        hist_df = pd.DataFrame(st.session_state["prediction_history"])
        st.dataframe(hist_df.tail(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -------- TAB 2: FEATURE IMPORTANCE --------
# -------- TAB 2: FEATURE IMPORTANCE --------
with tab2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("📈 Feature Importance (from ML model)")

    if model is None:
        st.error("Model file not found. Cannot compute feature importance.")
    else:
        try:
            # Your saved model is a plain XGBRegressor (not a Pipeline)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                n_features = len(importances)

                # Expected feature names (if you trained on raw columns directly)
                default_feature_names = [
                    "cylinders",
                    "displacement",
                    "horsepower",
                    "weight",
                    "acceleration",
                    "model year",
                    "origin"
                ]

                if n_features == len(default_feature_names):
                    feature_names = default_feature_names
                else:
                    # Fallback in case shape is different
                    feature_names = [f"Feature_{i}" for i in range(n_features)]

                fi_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False)

                st.write("Model thinks these features are most important for predicting mileage:")
                st.dataframe(fi_df, use_container_width=True)

                st.bar_chart(fi_df.set_index("Feature"))

                st.write(
                    "Higher importance means the feature has more influence on the mileage prediction.\n\n"
                    "Typically, features like **weight, displacement, horsepower and cylinders** "
                    "have a strong impact on fuel efficiency."
                )
            else:
                st.info("Current model does not expose `feature_importances_`.")
        except Exception as e:
            st.error("Could not compute feature importance due to a mismatch in model structure.")
            st.exception(e)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- TAB 3: HOW IT WORKS --------
with tab3:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("ℹ️ What does this app do?")

    st.write(
        """
        - This app predicts **car mileage** using a trained **Machine Learning model**.  
        - It takes common car specifications as input and outputs the expected **MPG** and equivalent **km/l**.  
        - It also estimates monthly & yearly **fuel cost** based on your running and fuel price.  
        - You can even **compare two cars (A vs B)** side-by-side.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.subheader("🧠 How does it work internally?")

    st.write(
        """
        1. Data cleaning (handling missing values like horsepower).  
        2. Outlier removal using statistical methods (IQR).  
        3. Feature engineering – scaling numeric features and encoding categorical variables via **scikit-learn Pipelines**.  
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
        - **Car buyers**: Rough idea of mileage and fuel cost based on specs.  
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
