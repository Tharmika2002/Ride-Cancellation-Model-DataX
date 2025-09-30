import streamlit as st
import joblib, json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# === Paths ===
MODEL_PATH = "final_model_pipeline.pkl"
LOOKUP_PATH = "lookup_stats.json"   # <-- fixed to match training export

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
stats = load_lookup()

# Helper to read class labels in correct order (for probability chart)
def get_classes(model):
    if hasattr(model, "classes_") and model.classes_ is not None:
        return list(model.classes_)
    clf = getattr(getattr(model, "named_steps", {}), "get", lambda k: None)("classifier")
    if clf is not None and hasattr(clf, "classes_"):
        return list(clf.classes_)
    return None

classes_ = get_classes(model)

st.title("🚖 Ride Booking Outcome Predictor (DataX)")

st.header("Enter Booking Details")

col1, col2 = st.columns(2)
with col1:
    date = st.date_input("Booking Date")
    time = st.time_input("Booking Time")
    # Use same label strings users know, but we'll map to training columns
    pickup = st.text_input("Pickup Area (e.g., Area-1)")
    customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.0, step=0.1)
    vehicle_type = st.selectbox("Vehicle Type", ["Auto", "Mini", "Sedan", "SUV", "Bike"])
with col2:
    drop = st.text_input("Drop Area (e.g., Area-10)")
    driver_rating = st.slider("Driver Rating", 1.0, 5.0, 4.0, step=0.1)
    payment_method = st.selectbox("Payment Method", ["Cash", "Card", "Wallet", "UPI"])

# Time features
dt = datetime.combine(date, time)
hour = dt.hour
day_of_week = dt.weekday()
is_weekend = int(day_of_week in (5, 6))
def assign_time_band(h):
    if 5 <= h <= 11: return "Morning"
    if 12 <= h <= 16: return "Afternoon"
    if 17 <= h <= 21: return "Evening"
    return "Night"
time_band = assign_time_band(hour)

# Lookup engineered features (computed on train split during training)
def get_lookup_rates(pu, dr, stats):
    pu_rate = stats["pickup_cancel_rate"].get(pu, stats["defaults"]["pickup_cancel_rate"])
    dr_rate = stats["drop_cancel_rate"].get(dr, stats["defaults"]["drop_cancel_rate"])
    pair_key = f"{pu}|||{dr}"
    pair_f = stats["pair_freq"].get(pair_key, stats["defaults"]["pair_freq"])
    return pu_rate, dr_rate, pair_f

pickup_cancel_rate, drop_cancel_rate, pickup_drop_pair_freq = get_lookup_rates(pickup, drop, stats)

# === Build single-row input matching the training schema EXACTLY ===
X_input = pd.DataFrame([{
    # numeric (same names as training)
    "hour_of_day": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "pickup_cancel_rate": pickup_cancel_rate,
    "drop_cancel_rate": drop_cancel_rate,
    "pickup_drop_pair_freq": pickup_drop_pair_freq,
    "customer_rating": float(customer_rating),
    "driver_rating": float(driver_rating),
    # categorical (same names as training)
    "time_band": time_band,
    "Pickup Location": pickup,
    "Drop Location": drop,
    "vehicle_type": vehicle_type,
    "payment_method": payment_method,
    # If your model did NOT use 'package_type', we do not include it.
}])

st.subheader("Prediction")
if st.button("Predict Booking Status"):
    try:
        pred = model.predict(X_input)[0]
        st.success(f"✅ Predicted Booking Status: **{pred}**")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_input)[0]
            labels = classes_ if classes_ is not None and len(classes_) == len(probs) else [str(i) for i in range(len(probs))]
            fig, ax = plt.subplots()
            ax.bar(labels, probs)
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.set_title("Prediction Confidence")
            plt.xticks(rotation=20, ha="right")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Make sure the model was trained with these exact feature names.")
