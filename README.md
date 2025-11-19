# 🚗 Auto MPG Prediction – Machine Learning & Streamlit App

This project predicts the **Miles Per Gallon (MPG)** of a car based on its technical specifications using 
Multiple Regression models (Linear, Lasso, Ridge, XGBoost).

The project includes:
- Data preprocessing & EDA
- Outlier detection and removal
- Model training & evaluation
- Best model saved as a `.pkl` file
- Streamlit web app for real-time MPG prediction

---

## 📂 Project Structure

```text
auto-mpg-app/
├─ app.py                     # Streamlit web app
├─ mpg_prediction_model.pkl   # Trained ML model (Pipeline with preprocessing)
├─ Auto_Mpg_Project_[LR].ipynb  # Jupyter/Colab notebook (training)
├─ requirements.txt
└─ README.md

⚙️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

XGBoost

Streamlit

🚀 How to Run the App Locally

Clone the repository or download the folder.

Create a virtual environment (optional but recommended).

Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


The app will open in your browser (usually at http://localhost:8501).


