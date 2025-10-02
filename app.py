import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime as dt
from pathlib import Path

# ==============================
# Page & Theme
# ==============================
st.set_page_config(page_title="Ride Cancellation Predictor — Random Forest", layout="centered")

# ----- Luca Davinci palette (from your image)
WARM_YELLOW = "#FEBA53"  # background
BROWN       = "#8A5438"
DARK_NAVY   = "#081A2D"
DEEP_BLUE   = "#05355D"  # primary action
BOLD_BLUE   = "#15608B"  # headings / accents
LIGHT_BLUE  = "#549ABE"  # input block background
WHITE       = "#FFFFFF"

st.markdown(
    f"""
    <style>
    /* Font (fallbacks if Google fails) */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;700&display=swap');

    :root {{
      --warm-yellow: {WARM_YELLOW};
      --brown: {BROWN};
      --dark-navy: {DARK_NAVY};
      --deep-blue: {DEEP_BLUE};
      --bold-blue: {BOLD_BLUE};
      --light-blue: {LIGHT_BLUE};
      --white: {WHITE};
    }}

    /* App background */
    .stApp {{
      background: var(--warm-yellow);
      background-attachment: fixed;
    }}

    /* Global text color tuning for contrast on yellow bg */
    .stApp, .stApp p, .stApp label, .stApp li, .stApp span {{
      color: #0d1b2a;
      font-weight: 500;
    }}

    /* Title */
    .big-title {{
      font-family: 'Poppins', system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      font-size: 2.2rem;
      font-weight: 700;
      text-align: center;
      color: var(--bold-blue);
      letter-spacing: .3px;
      margin: .25rem 0 .5rem 0;
    }}

    .subtitle {{
      color: #1f3147cc;
      text-align: center;
      margin-bottom: 1.25rem;
    }}

    /* Card/Section shells */
    .section {{
      border: 0;
      border-radius: 16px;
      padding: 18px 18px 14px 18px;
      background: var(--dark-navy);   /* form wrapper "another color" */
      box-shadow: 0 6px 20px rgba(0,0,0,.15);
    }}

    /* Input panel with light-blue background in the center */
    .input-panel {{
      border-radius: 14px;
      padding: 14px;
      background: var(--light-blue); /* your light-blue */
      margin-top: 6px;
    }}

    /* Center the whole booking-details block */
    .centered-container {{
      max-width: 820px;
      margin: 0 auto;
    }}

    /* Make Streamlit widget labels readable on dark form bg */
    .section :is(label, .stMarkdown p, .stDateInput, .stTimeInput) * {{
      color: var(--white) !important;
    }}

    /* Make inputs white inside */
    .input-panel .stSelectbox div[data-baseweb="select"] > div,
    .input-panel .stTextInput input,
    .input-panel .stDateInput input,
    .input-panel .stTimeInput input {{
      background: var(--white) !important;
      color: #0d1b2a !important;
      border-radius: 10px !important;
      border: 1px solid #e7eef6 !important;
    }}

    /* Chips under "Why this prediction?" */
    .pill {{
      display:inline-block; padding:.25rem .6rem; border-radius:999px;
      background:#f5f8fb; margin:.25rem .35rem .35rem 0; font-size:.85rem; color:#243b53;
      border: 1px solid #e6eef7;
    }}
    ul.reason-list {{ margin:.5rem 0 0 1.2rem; }}
    ul.reason-list li {{ margin:.2rem 0; }}

    /* Buttons — scoped colors by section */
    /* 1) Landing CTA */
    #landing .stButton > button {{
      background: var(--deep-blue);
      color: var(--white);
      border-radius: 12px;
      border: none;
    }}

    /* 2) Predict button in inputs */
    #inputs-section .stForm .stButton > button {{
      background: var(--deep-blue); /* blue primary */
      color: var(--white);
      border-radius: 12px;
      border: none;
    }}

    /* 3) "Show confidence by outcome" button uses brown */
    #confidence-block .stButton > button {{
      background: var(--brown);
      color: var(--white);
      border-radius: 10px;
      border: none;
    }}

    /* 4) Action buttons at bottom: New Prediction (bold-blue), Back to Start (deep-blue) */
    #predicted-actions .stButton:nth-of-type(1) > button {{
      background: var(--bold-blue);
      color: var(--white);
      border-radius: 12px;
      border: none;
    }}
    #predicted-actions .stButton:nth-of-type(2) > button {{
      background: var(--deep-blue);
      color: var(--white);
      border-radius: 12px;
      border: none;
    }}

    /* Streamlit default success/warning/error colors kept */
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
            pair_freqs = {{ (r["Pickup Location"], r["Drop Location"]): r["pickup_drop_pair_freq"] for _, r in dff.iterrows() }}
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

# ---------- Human-friendly, rule-based explanation ----------
def reasons_from_rules(row: pd.Series):
    pros_success, pros_cancel = [], []

    # Payment method
    if row.get("payment_method") == "Cash":
        pros_cancel.append("Cash payment")
    elif row.get("payment_method") == "Card":
        pros_success.append("Card payment")

    # Time band / hour
    band = row.get("time_band")
    hour = row.get("hour_of_day")
    if band in ("Night",) or (hour is not None and (hour >= 22 or hour <= 5)):
        pros_cancel.append("Night-time booking")
    elif band in ("Morning", "Afternoon"):
        pros_success.append(f"{band} booking")

    # Weekend vs weekday
    if int(row.get("is_weekend", 0)) == 1:
        pros_cancel.append("Weekend booking")
    else:
        pros_success.append("Weekday booking")

    # Historical cancellation rates
    pk = float(row.get("pickup_cancel_rate", 0))
    dp = float(row.get("drop_cancel_rate", 0))
    if pk >= 0.15:
        pros_cancel.append("Higher cancellations near pickup area")
    else:
        pros_success.append("Lower cancellations near pickup area")
    if dp >= 0.15:
        pros_cancel.append("Higher cancellations near drop area")
    else:
        pros_success.append("Lower cancellations near drop area")

    # Route familiarity (pair frequency)
    pf = float(row.get("pickup_drop_pair_freq", 0))
    if pf >= 30:
        pros_success.append("Familiar pickup→drop route")
    else:
        pros_cancel.append("Less common pickup→drop route")

    # Vehicle type
    vt = row.get("vehicle_type")
    if vt in ("Mini", "Sedan"):
        pros_success.append(f"{vt} vehicle")
    elif vt in ("Bike", "Auto"):
        pros_cancel.append(f"{vt} vehicle")

    # Dedupe
    def dedupe(seq):
        seen, out = set(), []
        for s in seq:
            if s not in seen:
                out.append(s); seen.add(s)
        return out

    return dedupe(pros_success), dedupe(pros_cancel)

def pick_reasons_for_prediction(row_df: pd.DataFrame, predicted_label: str, top_k: int = 3):
    row = row_df.iloc[0]
    pros_success, pros_cancel = reasons_from_rules(row)
    is_success = "success" in predicted_label.lower()
    pool = pros_success if is_success else pros_cancel

    order = [
        "Higher cancellations near pickup area",
        "Higher cancellations near drop area",
        "Lower cancellations near pickup area",
        "Lower cancellations near drop area",
        "Night-time booking",
        "Weekend booking",
        "Morning booking",
        "Afternoon booking",
        "Familiar pickup→drop route",
        "Less common pickup→drop route",
        "Cash payment",
        "Card payment",
        "Bike vehicle",
        "Auto vehicle",
        "Mini vehicle",
        "Sedan vehicle",
        "Weekday booking",
    ]
    ranked = [r for r in order if r in pool] or pool
    reasons = ranked[:top_k]
    if is_success:
        reasons = [f"{r} **helped increase success**" for r in reasons]
    else:
        reasons = [f"{r} **increased cancellation risk**" for r in reasons]
    return reasons

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
st.markdown('<p class="subtitle">Predict booking status and see why the model thinks so.</p>', unsafe_allow_html=True)

# ==============================
# Landing: single CTA
# ==============================
if st.session_state.ui_stage == "landing":
    st.markdown('<div id="landing" class="section centered-container">', unsafe_allow_html=True)
    st.write("Click below to check your booking’s predicted status.")
    if st.button("🔎 Check your prediction", use_container_width=True):
        st.session_state.ui_stage = "inputs"
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Inputs form
# ==============================
if st.session_state.ui_stage == "inputs":
    st.markdown('<div id="inputs-section" class="section centered-container">', unsafe_allow_html=True)
    with st.form("inputs-form", clear_on_submit=False):
        st.markdown('<h3 style="color: var(--white); text-align:center; margin:0 0 10px;">📋 Booking details</h3>', unsafe_allow_html=True)

        # Light-blue panel that wraps inputs
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            pickup_location = st.selectbox("Pickup Location", AREAS, index=0)
            vehicle_type = st.selectbox("Vehicle Type", VEHICLES, index=0)
            payment_method = st.selectbox("Payment Method", PAYMENTS, index=1)
        with c2:
            drop_location = st.selectbox("Drop Location", AREAS, index=1)

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        dcol, tcol = st.columns(2)
        with dcol:
            booking_date = st.date_input("Date", value=dt.date.today())
        with tcol:
            booking_time = st.time_input("Time", value=dt.datetime.now().time())
        st.markdown('</div>', unsafe_allow_html=True)  # close input-panel

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
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Predicted view
# ==============================
if st.session_state.ui_stage == "predicted":
    input_df  = st.session_state.last_input_df
    time_feats = st.session_state.last_time_feats
    pred      = st.session_state.last_pred
    proba     = st.session_state.last_proba
    classes   = get_classes()

    # White card for prediction area (you asked for white “for this prediction”)
    with st.container():
        st.markdown('<div class="centered-container" style="background:white;border-radius:16px;padding:18px;box-shadow:0 6px 20px rgba(0,0,0,.12);">', unsafe_allow_html=True)

        st.markdown("### 🔮 Prediction")
        pred_lower = str(pred).lower()
        if "success" in pred_lower:
            st.success(f"✅ Predicted Booking Status: **{pred}**")
        elif "cancel" in pred_lower:
            st.error(f"❌ Predicted Booking Status: **{pred}**")
        else:
            st.warning(f"⚠️ Predicted Booking Status: **{pred}**")

        # --- Why this prediction?
        st.markdown("#### 🧠 Why this prediction?")
        chips = [
            f'<span class="pill">Hour: {time_feats["hour_of_day"]}</span>',
            f'<span class="pill">Day: {time_feats["day_name"]}</span>',
            f'<span class="pill">Band: {time_feats["time_band"]}</span>',
            f'<span class="pill">Pickup prior: {input_df["pickup_cancel_rate"].iloc[0]:.2f}</span>',
            f'<span class="pill">Drop prior: {input_df["drop_cancel_rate"].iloc[0]:.2f}</span>',
            f'<span class="pill">Route freq: {int(input_df["pickup_drop_pair_freq"].iloc[0])}</span>',
        ]
        st.markdown("".join(chips), unsafe_allow_html=True)

        reasons = pick_reasons_for_prediction(input_df, str(pred), top_k=3)
        st.markdown("<ul class='reason-list'>" + "".join([f"<li>{r}</li>" for r in reasons]) + "</ul>", unsafe_allow_html=True)

        st.divider()

        # --- Confidence chart toggle (brown button)
        st.markdown('<div id="confidence-block">', unsafe_allow_html=True)
        if st.button("Show confidence by outcome"):
            if proba is None:
                st.info("Confidence details aren’t available for this model.")
            else:
                prob_df = pd.DataFrame({"Outcome": classes, "Confidence": proba})
                st.bar_chart(prob_df.set_index("Outcome"))
                top_idx = int(np.argmax(proba))
                st.caption(f"The model is most confident about **{classes[top_idx]}** "
                           f"({100*float(np.max(proba)):.1f}%).")
        else:
            st.caption("Click to see the model’s confidence for each possible outcome.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # Bottom action buttons with two different blues
        st.markdown('<div id="predicted-actions">', unsafe_allow_html=True)
        cols = st.columns(2)
        if cols[0].button("← New prediction", use_container_width=True):
            st.session_state.ui_stage = "inputs"
            st.rerun()
        if cols[1].button("🏠 Back to start", use_container_width=True):
            st.session_state.ui_stage = "landing"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # close white card
