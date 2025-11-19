# 🚗 Car Mileage Prediction Web App (MPG → KM/L)

A fully deployed **Machine Learning + Streamlit Web Application** that predicts the **fuel efficiency (MPG & KM/L)** of a car based on technical specifications. The app also estimates **fuel cost per month & year**, and allows you to **compare two cars** realistically for Indian users.

🔗 **Live Demo (Optional):** *(https://auto-mpg-prediction-bepcnhvz6qd3t8h8qhp6e8.streamlit.app/#auto-mpg-ml-web-app)*  
👨‍💻 **Developed by:** **Dikesh Chavhan**

---

## 📌 Key Features

🔹 **Predict Mileage** (Miles Per Gallon + Kilometres per Litre)  
🔹 **Indian Fuel Cost Calculator** (Monthly & Yearly)  
🔹 **Compare Two Cars Side-by-Side**  
🔹 **Prediction History (Session-Based)**  
🔹 **Feature Importance Graph (XGBoost Model)**  
🔹 **Smart UI + Hindi Assistance** *(for better understanding)*  
🔹 **Responsive Modern UI with Google Fonts, Gradients & Animations*

---

## 🖼️ Screenshots (To Be Added)


| *(<img width="1527" height="677" alt="Screenshot 2025-11-19 180303" src="https://github.com/user-attachments/assets/5a819930-78db-4bdd-a34c-1805af3c2d88" />
)* | *(<img width="1668" height="712" alt="Screenshot 2025-11-19 180421" src="https://github.com/user-attachments/assets/3c5b1ab4-07e9-4f89-9b38-af147b59a27c" />
)* | *(<img width="1619" height="735" alt="Screenshot 2025-11-19 180459" src="https://github.com/user-attachments/assets/7b3583d0-176b-4c6e-95b3-1104a6c054c7" />
)* |

---

## 🧠 Machine Learning Details

**Dataset:** Auto MPG Dataset (UCI Repository)  
**Target:** `mpg` → Converted to **km/l**  
**Algorithms Used:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- **XGBoost (Best Model – Used in Deployment)**

**Preprocessing Techniques**
- Missing values handled
- Outlier treatment using IQR
- Feature Scaling (StandardScaler)
- One-Hot Encoding for categorical features  
- Model saved as **`mpg_prediction_model.pkl`**

---

## 🗂️ Tech Stack

| Category | Technologies |
|----------|--------------|
| ML | Scikit-learn, XGBoost, Pandas, NumPy |
| Deployment | Streamlit |
| Visualization | Matplotlib, Seaborn, Streamlit Charts |
| Backend Logic | Python |
| Styling | Custom CSS, Google Fonts (Poppins) |

---

## 🚀 How to Run Locally

### 📍 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/car-mileage-ml-app.git

📍 2. Navigate to the Project Folder
cd car-mileage-ml-app

📍 3. Install Dependencies
pip install -r requirements.txt

📍 4. Run the App
streamlit run app.py
---
☁️ Deployment Guide (Streamlit Cloud)

Create a GitHub repository and upload:

app.py

mpg_prediction_model.pkl

requirements.txt

README.md

Go to: https://share.streamlit.io/

Connect your GitHub repository

Select:

Main File: app.py

Click Deploy 🚀
