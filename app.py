import streamlit as st
import joblib
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Paths ===
MODEL_PATH = "final_model_pipeline.pkl"
LOOKUP_PATH = "lookup_stats.json"

# === Load model & lookup stats ===
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Please add final_model_pipeline.pkl to the repo.")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_lookup():
    if not os.path.exists(LOOKUP_PATH):
        st.error("❌ Lookup stats not found. Please add models/lookup_stats.json to the repo.")
        st.stop()
    with open(LOOKUP_PATH, "r") as f:
        return json.load(f)

model = load_pipeline()
lookup_stats = load_lookup()

st.title("🚖 Ride Booking Outcome Predictor")

# === User Inputs ===
st.header("Enter Booking Details")

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Booking Date")
    time = st.time_input("Booking Time")
    pickup = st.text_input("Pickup Area (e.g., Area-1)")
    drop = st.text_input("Drop Area (e.g., Area-10)")
    vehicle = st.selectbox("Vehicle Type", ["Auto", "Mini", "Sedan", "SUV"])
with col2:
    payment = st.selectbox("Payment Method", ["Cash", "UPI", "Wallet", "Card"])
    package = st.selectbox("Package Type", ["Local", "Rental", "Outstation"])
    cust_rating = st.slider("Customer Rating", 1.0, 5.0, 4.0)
    driver_rating = st.slider("Driver Rating", 1.0, 5.0, 4.0)

# === Feature Engineering (light) ===
datetime_str = str(date) + " " + str(time)
dt = pd.to_datetime(datetime_str)

hour = dt.hour
day_of_week = dt.dayofweek
is_weekend = int(day_of_week in [5, 6])

def assign_time_band(h):
    if 5 <= h <= 11: return "Morning"
    elif 12 <= h <= 16: return "Afternoon"
    elif 17 <= h <= 21: return "Evening"
    else: return "Night"

time_band = assign_time_band(hour)

# === Construct feature row ===
row = {
    "Pickup": pickup,
    "Drop": drop,
    "Vehicle_Type": vehicle,
    "Payment_Method": payment,
    "Package_Type": package,
    "Customer_Rating": cust_rating,
    "Driver_Rating": driver_rating,
    "hour_of_day": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "time_band": time_band,
}

X_input = pd.DataFrame([row])

# === Prediction ===
if st.button("Predict Booking Status"):
    pred = model.predict(X_input)[0]
    probs = model.predict_proba(X_input)[0]

    st.success(f"✅ Predicted Booking Status: **{pred}**")

    # Probability chart
    fig, ax = plt.subplots()
    ax.bar(model.classes_, probs, color="skyblue")
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)
