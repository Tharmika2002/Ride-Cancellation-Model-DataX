# app.py
import os
import sys
import json
import types
import hashlib
from datetime import datetime
from typing import List, Tuple

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import custom_transformers  # must come first to satisfy pickle

# ---------- Version banner ----------
import sklearn  # noqa: F401
st.caption(
    f"Python {sys.version.split()[0]} | "
    f"sklearn {sklearn.__version__} | "
    f"numpy {np.__version__} | "
    f"pandas {pd.__version__}"
)

# ---------- Paths ----------
MODEL_PATH = "final_model_pipeline.pkl"
LOOKUP_CANDIDATES = ["lookup_stats.json", "models/lookup_stats.json"]

# ---------- Helpers ----------
def _first_existing(paths: List[str]):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _file_info(path: str) -> str:
    size = os.path.getsize(path) if os.path.exists(path) else -1
    sha = ""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        sha = h.hexdigest()[:16]
    except Exception:
        pass
    return f"{size} bytes | sha256:{sha}"

# ---------- Loaders ----------
@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file not found. Add final_model_pipeline.pkl to repo root.")
        st.stop()
    st.write(f"**Model file:** `{MODEL_PATH}` ({_file_info(MODEL_PATH)})")
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

@st.cache_data
def load_lookup():
    path = _first_existing(LOOKUP_CANDIDATES)
    if not path:
        st.error("❌ Lookup stats not found. Add lookup_stats.json at root or models/ folder.")
        st.stop()
    with open(path, "r") as f:
        return json.load(f)

# ---------- Init ----------
model = load_pipeline()
stats = load_lookup()

def get_classes(m):
    if hasattr(m, "classes_") and m.classes_ is not None:
        return list(m.classes_)
    steps = getattr(m, "named_steps", None)
    if isinstance(steps, dict):
        clf = steps.get("classifier")
        if clf is not None and hasattr(clf, "classes_"):
            return list(clf.classes_)
    return None

classes_ = get_classes(model)

# ---------- Streamlit UI ----------
st.title("🚖 Ride Booking Outcome Predictor (DataX)")
st.header("Enter Booking Details")

col1, col2 = st.columns(2)
with col1:
    date = st.date_input("Booking Date")
    time = st.time_input("Booking Time")
    pickup = st.text_input("Pickup Area (e.g., Area-1)")
    customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.0, step=0.1)
    vehicle_type = st.selectbox("Vehicle Type", ["Auto", "Mini", "Sedan", "SUV", "Bike"])
with col2:
    drop = st.text_input("Drop Area (e.g., Area-10)")
    driver_rating = st.slider("Driver Rating", 1.0, 5.0, 4.0, step=0.1)
    payment_method = st.selectbox("Payment Method", ["Cash", "Card", "Wallet", "UPI"])

# ---------- Feature Engineering ----------
dt = datetime.combine(date, time)
hour = dt.hour
day_of_week = dt.weekday()
is_weekend = int(day_of_week in (5, 6))

def assign_time_band(h):
    if 5 <= h <= 11:
        return "Morning"
    if 12 <= h <= 16:
        return "Afternoon"
    if 17 <= h <= 21:
        return "Evening"
    return "Night"

time_band = assign_time_band(hour)

def get_lookup_rates(pu, dr, _stats):
    pu_rate = _stats["pickup_cancel_rate"].get(pu, _stats["defaults"]["pickup_cancel_rate"])
    dr_rate = _stats["drop_cancel_rate"].get(dr, _stats["defaults"]["drop_cancel_rate"])
    pair_key = f"{pu}|||{dr}"
    pair_f = _stats["pair_freq"].get(pair_key, _stats["defaults"]["pair_freq"])
    return pu_rate, dr_rate, pair_f

pickup_cancel_rate, drop_cancel_rate, pickup_drop_pair_freq = get_lookup_rates(pickup, drop, stats)

X_input = pd.DataFrame([{
    # Numeric features
    "hour_of_day": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "pickup_cancel_rate": pickup_cancel_rate,
    "drop_cancel_rate": drop_cancel_rate,
    "pickup_drop_pair_freq": pickup_drop_pair_freq,
    "customer_rating": float(customer_rating),
    "driver_rating": float(driver_rating),
    # Categorical features
    "time_band": time_band,
    "Pickup Location": pickup,
    "Drop Location": drop,
    "vehicle_type": vehicle_type,
    "payment_method": payment_method,
}])

st.subheader("Input Features")
st.dataframe(X_input)

# ---------- Prediction ----------
st.subheader("Prediction")
if pickup and drop:
    if st.button("Predict Booking Status"):
        try:
            pred = model.predict(X_input)[0]
            st.success(f"✅ Predicted Booking Status: **{pred}**")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_input)[0]
                labels = classes_ or [str(i) for i in range(len(probs))]

                fig, ax = plt.subplots()
                ax.bar(labels, probs, color="#4c78a8")
                ax.set_ylabel("Probability")
                ax.set_ylim(0, 1)
                ax.set_title("Prediction Confidence")
                plt.xticks(rotation=20, ha="right")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Ensure feature names match training and custom transformers are implemented.")
else:
    st.info("Please enter both Pickup and Drop locations to enable prediction.")
