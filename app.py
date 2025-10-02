import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime as dt
from pathlib import Path

# ==============================
# Page & theme
# ==============================
st.set_page_config(page_title="Ride Cancellation Predictor — Random Forest", layout="centered")
st.markdown(
    """
    <style>
    .big-title { font-size: 2rem; font-weight: 700; margin-bottom: .25rem; }
    .subtitle { color: #666; margin-bottom: 1.5rem; }
    .pill { display:inline-block; padding: .25rem .6rem; border-radius: 999px; background:#f5f5f7; margin-right:.4rem; font-size:.85rem; }
    .muted { color:#777; font-size:.9rem; }
    .section { border:1px solid #eee; border-radius: 12px; padding: 16px; background: #fff; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# Model loader (Pipeline OR dict bundle)
# ==============================
MODEL_PATH = "random_forest_smote_model.pkl"

@st.cache_resource
def load_bundle(path: str):
    obj = joblib.load(path)
    if hasattr(obj, "predict"):  # sklearn Pipeline
        pre = getattr(obj, "named_steps", {}).get("preprocessor", None)
        clf = getattr(obj, "named_steps", {}).get("classifier", None)
        return {"bundle_type": "pipeline", "pipeline": obj, "pre": pre, "clf": clf}
    elif isinstance(obj, dict) and "model" in obj:  # dict bundle
        return {"bundle_type": "dict", "pipeline": None, "pre": obj.get("preprocessor"), "clf": obj["model"], "feature_order": obj.get("feature_order")}
    else:
        raise ValueError("Unsupported model bundle format.")

bundle = load_bundle(MODEL_PATH)
pre = bundle["pre"]
clf = bundle["clf"]

def get_classes():
    if bundle["bundle_type"] == "pipeline":
        return bundle["pipeline"].classes_
    return clf.classes_

def predict_df(df: pd.DataFrame):
    if bundle["bundle_type"] == "pipeline":
        return bundle["pipeline"].predict(df)
    Xp = pre.transform(df) if pre is not None else df
    return clf.predict(Xp)

def predict_proba_df(df: pd.DataFrame):
    if bundle["bundle_type"] == "pipeline":
        return bundle["pipeline"].predict_proba(df) if hasattr(bundle["pipeline"], "predict_proba") else None
    Xp = pre.transform(df) if pre is not None else df
    return clf.predict_proba(Xp) if hasattr(clf, "predict_proba") else None

# ==============================
# Priors & frequencies (CSV-if-available, else fallback)
# ==============================
pickup_priors_path = Path("pickup_priors.csv")
drop_priors_path   = Path("drop_priors.csv")
pair_freqs_path    = Path("pair_freqs.csv")

pickup_priors, drop_priors, pair_freqs = {}, {}, {}

try:
    if pickup_priors_path.exists():
        dfp = pd.read_csv(pickup_priors_path)
        if {"Pickup Location", "pickup_cancel_rate"}.issubset(dfp.columns):
            pickup_priors = dict(zip(dfp["Pickup Location"], dfp["pickup_cancel_rate"]))
except Exception:
    pickup_priors = {}

try:
    if drop_priors_path.exists():
        dfd = pd.read_csv(drop_priors_path)
        if {"Drop Location", "drop_cancel_rate"}.issubset(dfd.columns):
            drop_priors = dict(zip(dfd["Drop Location"], dfd["drop_cancel_rate"]))
except Exception:
    drop_priors = {}

try:
    if pair_freqs_path.exists():
        dff = pd.read_csv(pair_freqs_path)
        req = {"Pickup Location", "Drop Location", "pickup_drop_pair_freq"}
        if req.issubset(dff.columns):
            pair_freqs = { (r["Pickup Location"], r["Drop Location"]): r["pickup_drop_pair_freq"] for _, r in dff.iterrows() }
except Exception:
    pair_freqs = {}

# ==============================
# Helpers
# ==============================
AREAS = [f"Area-{i}" for i in range(1, 51)]
VEHICLES = ["Auto", "Mini", "Sedan", "Bike"]
PAYMENTS = ["Cash", "Card"]
DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def derive_time_features(ts: dt.datetime):
    hour = ts.hour
    dow = ts.weekday()
    weekend = 1 if dow >= 5 else 0
    if 5 <= hour <= 11: band = "Morning"
    elif 12 <= hour <= 16: band = "Afternoon"
    elif 17 <= hour <= 21: band = "Evening"
    else: band = "Night"
    return {"hour_of_day": hour, "day_of_week": dow, "is_weekend": weekend, "time_band": band, "day_name": DAY_NAMES[dow]}

def get_pair_freq(pickup, drop):
    if (pickup, drop) in pair_freqs:
        try: return float(pair_freqs[(pickup, drop)])
        except Exception: pass
    return 50.0 if pickup == drop else 10.0

def get_pickup_prior(pickup, time_band):
    if pickup in pickup_priors:
        try: return float(pickup_priors[pickup])
        except Exception: pass
    return 0.25 if time_band == "Night" else 0.10

def get_drop_prior(drop, time_band):
    if drop in drop_priors:
        try: return float(drop_priors[drop])
        except Exception: pass
    return 0.20 if time_band == "Night" else 0.08

def build_input_df(booking_dt, pickup_location, drop_location, vehicle_type, payment_method):
    tf = derive_time_features(booking_dt)
    row = {
        "hour_of_day": tf["hour_of_day"],
        "day_of_week": tf["day_of_week"],
        "is_weekend": tf["is_weekend"],
        "pickup_cancel_rate": get_pickup_prior(pickup_location, tf["time_band"]),
        "drop_cancel_rate": get_drop_prior(drop_location, tf["time_band"]),
        "pickup_drop_pair_freq": get_pair_freq(pickup_location, drop_location),
        "time_band": tf["time_band"],
        "Pickup Location": pickup_location,
        "Drop Location": drop_location,
        "vehicle_type": vehicle_type,
        "payment_method": payment_method,
    }
    return pd.DataFrame([row]), tf

# ---------- Human-friendly explanation (no SHAP) ----------
def _prettify_feat_name(name: str) -> str:
    return (name.replace("num__", "")
                .replace("cat__encoder__", "")
                .replace("pickup_cancel_rate", "pickup-area cancel rate")
                .replace("drop_cancel_rate", "drop-area cancel rate")
                .replace("pickup_drop_pair_freq", "route frequency")
                .replace("hour_of_day", "hour of day")
                .replace("day_of_week", "day of week")
                .replace("is_weekend", "weekend"))

def _humanize_onehot(name: str) -> str:
    # Turn e.g. 'payment method_Cash' into 'payment method is Cash'
    if "_" not in name:
        return name
    base, val = name.split("_", 1)
    base = (base.replace("Pickup Location", "pickup location")
                .replace("Drop Location", "drop location")
                .replace("payment_method", "payment method")
                .replace("vehicle_type", "vehicle type")
                .replace("time_band", "time band"))
    return f"{base} is {val}"

def _humanize_numeric(base_col: str, raw_val):
    if base_col == "pickup_cancel_rate":
        return "higher pickup-area cancellation rate" if float(raw_val) >= 0.15 else "lower pickup-area cancellation rate"
    if base_col == "drop_cancel_rate":
        return "higher drop-area cancellation rate" if float(raw_val) >= 0.15 else "lower drop-area cancellation rate"
    if base_col == "pickup_drop_pair_freq":
        return "a common route" if float(raw_val) >= 30 else "a relatively rare route"
    if base_col == "hour_of_day":
        return f"the booking hour (~{int(raw_val):02d}:00)"
    if base_col == "day_of_week":
        idx = int(raw_val) if 0 <= int(raw_val) < 7 else int(raw_val) % 7
        return f"the day ({DAY_NAMES[idx]})"
    if base_col == "is_weekend":
        return "it being a weekend" if int(raw_val) == 1 else "it being a weekday"
    return f"{base_col} ≈ {raw_val}"

def explain_with_importances(input_df: pd.DataFrame, pre, clf, top_k: int = 3) -> str:
    try:
        if pre is None or not hasattr(clf, "feature_importances_"):
            return "A short explanation isn’t available for this model."

        Xtr = pre.transform(input_df)
        x = np.asarray(Xtr.toarray()[0]) if hasattr(Xtr, "toarray") else np.asarray(Xtr[0])
        importances = np.asarray(clf.feature_importances_)
        if importances.shape[0] != x.shape[0]:
            return "Could not align features for explanation."

        contrib = x * importances
        order = np.argsort(np.abs(contrib))[::-1][:top_k]

        try:
            names = pre.get_feature_names_out()
        except Exception:
            names = np.array([f"feature_{i}" for i in range(len(importances))])

        bits = []
        for idx in order:
            raw_name = str(names[idx])
            nice = _prettify_feat_name(raw_name)

            if raw_name.startswith("num__"):
                base_col = raw_name.split("__", 1)[1]
                raw_val = input_df.iloc[0].get(base_col, None)
                phrase = _humanize_numeric(base_col, raw_val)
            else:
                phrase = _humanize_onehot(nice)

            inc = contrib[idx] > 0
            direction_text = "increased the chance of this outcome" if inc else "reduced the chance of this outcome"

            # For one-hot, only mention the active category
            if raw_name.startswith("cat__") and x[idx] <= 0.5:
                continue

            bits.append(f"{phrase} {direction_text}")

        if not bits:
            return "The model combined several weaker signals; no single factor dominated."

        if len(bits) == 1:
            return f"This prediction was mainly driven by {bits[0]}."
        if len(bits) == 2:
            return f"This prediction was mainly driven by {bits[0]} and {bits[1]}."
        return f"This prediction was mainly driven by {', '.join(bits[:-1])}, and {bits[-1]}."
    except Exception:
        return "Could not generate a short explanation for this prediction."

# ==============================
# Session State
# ==============================
if "ui_stage" not in st.session_state:
    st.session_state.ui_stage = "landing"
if "last_input_df" not in st.session_state:
    st.session_state.last_input_df = None
if "last_time_feats" not in st.session_state:
    st.session_state.last_time_feats = None
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None
if "last_proba" not in st.session_state:
    st.session_state.last_proba = None

# ==============================
# Header
# ==============================
st.markdown('<div class="big-title">🚖 Ride Cancellation Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Random Forest • Simple. Clear. Explainable.</div>', unsafe_allow_html=True)

# ==============================
# Landing: single CTA
# ==============================
if st.session_state.ui_stage == "landing":
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.write("Click below to check your booking’s predicted status.")
    if st.button("🔎 Check your prediction", type="primary", use_container_width=True):
        st.session_state.ui_stage = "inputs"
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Inputs form
# ==============================
if st.session_state.ui_stage == "inputs":
    with st.form("inputs-form", clear_on_submit=False):
        st.markdown("### 📋 Booking details")
        c1, c2 = st.columns(2)
        with c1:
            pickup_location = st.selectbox("Pickup Location", AREAS, index=0)
            vehicle_type = st.selectbox("Vehicle Type", VEHICLES, index=0)
            payment_method = st.selectbox("Payment Method", PAYMENTS, index=1)
        with c2:
            drop_location = st.selectbox("Drop Location", AREAS, index=1)

        st.markdown("#### 🗓️ Date & Time")
        dcol, tcol = st.columns([1,1])
        with dcol:
            booking_date = st.date_input("Date", value=dt.date.today())
        with tcol:
            booking_time = st.time_input("Time", value=dt.datetime.now().time())

        submitted = st.form_submit_button("✨ Predict ride status", use_container_width=True)
        if submitted:
            booking_dt = dt.datetime.combine(booking_date, booking_time)
            input_df, time_feats = build_input_df(
                booking_dt=booking_dt,
                pickup_location=pickup_location,
                drop_location=drop_location,
                vehicle_type=vehicle_type,
                payment_method=payment_method,
            )
            pred = predict_df(input_df)[0]
            try:
                proba = predict_proba_df(input_df)[0]
            except Exception:
                proba = None

            st.session_state.last_input_df = input_df
            st.session_state.last_time_feats = time_feats
            st.session_state.last_pred = pred
            st.session_state.last_proba = proba
            st.session_state.ui_stage = "predicted"
            st.experimental_rerun()

# ==============================
# Predicted view
# ==============================
if st.session_state.ui_stage == "predicted":
    input_df = st.session_state.last_input_df
    time_feats = st.session_state.last_time_feats
    pred = st.session_state.last_pred
    proba = st.session_state.last_proba
    classes = get_classes()

    st.markdown("### 🔮 Prediction")
    pred_lower = str(pred).lower()
    if "success" in pred_lower:
        st.success(f"✅ Predicted Booking Status: **{pred}**")
    elif "cancel" in pred_lower:
        st.error(f"❌ Predicted Booking Status: **{pred}**")
    else:
        st.warning(f"⚠️ Predicted Booking Status: **{pred}**")

    # --- Why this prediction? (chips + short readable text)
    with st.container():
        st.markdown("#### 🧠 Why this prediction?")
        chips = [
            f'<span class="pill">Hour: {time_feats["hour_of_day"]}</span>',
            f'<span class="pill">Day: {time_feats["day_name"]}</span>',
            f'<span class="pill">Band: {time_feats["time_band"]}</span>',
            f'<span class="pill">Pickup prior: {input_df["pickup_cancel_rate"].iloc[0]:.2f}</span>',
            f'<span class="pill">Drop prior: {input_df["drop_cancel_rate"].iloc[0]:.2f}</span>',
            f'<span class="pill">Route freq: {int(input_df["pickup_drop_pair_freq"].iloc[0])}</span>',
        ]
        st.markdown(" ".join(chips), unsafe_allow_html=True)
        explanation = explain_with_importances(input_df, pre, clf, top_k=3)
        st.write(explanation)

    st.divider()

    # === Only visualization we keep: Class probabilities (on demand) ===
    st.markdown("### 📊 Class probability")
    if st.button("Show class probabilities"):
        if proba is None:
            st.info("Model does not expose probabilities.")
        else:
            prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
            st.bar_chart(prob_df.set_index("Class"))
            top_idx = int(np.argmax(proba))
            st.caption(f"Top class: **{classes[top_idx]}** with {100*float(np.max(proba)):.1f}% confidence.")
    else:
        st.caption("Click the button to see the model’s class probabilities.")

    st.divider()
    cols = st.columns(2)
    if cols[0].button("← New prediction", use_container_width=True):
        st.session_state.ui_stage = "inputs"
        st.experimental_rerun()
    if cols[1].button("🏠 Back to start", use_container_width=True):
        st.session_state.ui_stage = "landing"
        st.experimental_rerun()
