# ==========================================
# Crop Recommendation System (Pro Version)
# Author: Arvind Singh Jhala
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Crop AI", page_icon="🌱", layout="centered")

# -------------------------------
# CUSTOM CSS (Premium UI)
# -------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
h1 {
    text-align: center;
    color: #00ffcc;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 15px rgba(0,255,204,0.2);
}
.stButton>button {
    background-color: #00ffcc;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOGIN SYSTEM
# -------------------------------
def login():
    st.title("🔐 Login to Crop AI")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.success("Login successful 🚀")
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("crop_recommendation_model.pkl")

# -------------------------------
# TITLE
# -------------------------------
st.markdown("<h1>🌱 Crop Recommendation AI</h1>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("### Enter Soil & Climate Details")

# -------------------------------
# INPUTS (2 columns)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("🌿 Nitrogen", min_value=0.0)
    phosphorus = st.number_input("🧪 Phosphorus", min_value=0.0)
    potassium = st.number_input("🧱 Potassium", min_value=0.0)
    temperature = st.number_input("🌡️ Temperature (°C)")

with col2:
    humidity = st.number_input("💧 Humidity (%)")
    ph = st.number_input("⚗️ pH Value")
    rainfall = st.number_input("🌧️ Rainfall (mm)")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Best Crops"):

    with st.spinner("Analyzing soil... 🌍"):
        input_data = pd.DataFrame({
            'Nitrogen': [nitrogen],
            'Phosphorus': [phosphorus],
            'Potassium': [potassium],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'pH_Value': [ph],
            'Rainfall': [rainfall]
        })

        probabilities = model.predict_proba(input_data)[0]
        top3_indices = probabilities.argsort()[-3:][::-1]
        top3_crops = model.classes_[top3_indices]
        top3_probs = probabilities[top3_indices]

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write("### 🌾 Top Recommendations")

    for crop, prob in zip(top3_crops, top3_probs):
        st.success(f"{crop} → {prob*100:.2f}% confidence")

    # -------------------------------
    # CHART (IMPORTANT)
    # -------------------------------
    st.write("### 📊 Confidence Chart")

    fig, ax = plt.subplots()
    ax.bar(top3_crops, top3_probs)
    ax.set_ylabel("Confidence")
    ax.set_title("Top Crop Predictions")

    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<hr>
<center>🚀 Built by Arvind Singh Jhala | AIML Student</center>
""", unsafe_allow_html=True)
