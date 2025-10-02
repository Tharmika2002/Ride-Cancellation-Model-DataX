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
    ul.reason-list { margin: .5rem 0 0 1.2rem; }
    ul.reason-list li { margin: .15rem 0; }
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

def _humanize_onehot(nice_name: str):
    """
    Turn one-hot like 'payment method_Cash' into ('Cash payment', 'factor text').
    We keep only user-friendly categories; skip Area-* categories.
    """
    if "_" not in nice_name:
        return None
    base, val = nice_name.split("_", 1)
    base_norm = (base.replace("payment_method", "payment method")
                      .replace("vehicle_type", "vehicle type")
                      .replace("time_band", "time band")
                      .strip())

    if base_norm == "payment method":
        return "Cash payment" if val == "Cash" else "Card payment"
    if base_norm == "vehicle type":
        return f"{val} vehicle"
    if base_norm == "time band":
        return f"{val.capitalize()} time"
    return None  # skip others to avoid awkward text

def _direction_phrase(inc: bool, predicted_label: str):
    """
    Map direction to human phrases depending on predicted class.
    """
    is_success = "success" in predicted_label.lower()
    if is_success:
        return "helped **increase success**" if inc else "helped **reduce success**"
    else:
        return "helped **increase cancellation risk**" if inc else "helped **reduce cancellation risk**"

def _humanize_numeric(base_col: str, raw_val):
    """
    Human title for numeric features using ORIGINAL value (no scaled numbers).
    """
    try:
        v = float(raw_val)
    except Exception:
        v = None

    if base_col == "pickup_cancel_rate":
        return "Higher cancellations near pickup" if (v is not None and v >= 0.15) else "Lower cancellations near pickup"
    if base_col == "drop_cancel_rate":
        return "Higher cancellations near drop" if (v is not None and v >= 0.15) else "Lower cancellations near drop"
    if base_col == "pickup_drop_pair_freq":
        # Do not assert common/rare effects; keep neutral wording
        return "This pickup→drop route pattern"
    if base_col == "hour_of_day":
        return f"Booking around {int(v):02d}:00" if v is not None else "Booking time"
    if base_col == "day_of_week":
        try:
            return f"Booking on {DAY_NAMES[int(v) % 7]}"
        except Exception:
            return "Day of week"
    if base_col == "is_weekend":
        return "Weekend booking" if int(v) == 1 else "Weekday booking"
    return base_col

def explain_with_importances(input_df: pd.DataFrame, pre, clf, predicted_label: str, top_k: int = 3):
    """
    Produces a short bullet list of human-friendly reasons.
    Uses RF global importances as a local proxy and avoids awkward claims.
    """
    try:
        if pre is None or not hasattr(clf, "feature_importances_"):
            return ["A short explanation isn’t available for this model."]

        Xtr = pre.transform(input_df)
        x = np.asarray(Xtr.toarray()[0]) if hasattr(Xtr, "toarray") else np.asarray(Xtr[0])
        importances = np.asarray(clf.feature_importances_)
        if importances.shape[0] != x.shape[0]:
            return ["The model combined several signals for this prediction."]

        contrib = x * importances
        order = np.argsort(np.abs(contrib))[::-1]

        try:
            names = pre.get_feature_names_out()
        except Exception:
            names = np.array([f"feature_{i}" for i in range(len(importances))])

        reasons = []
        for idx in order:
            if len(reasons) >= top_k:
                break

            raw_name = str(names[idx])
            inc = bool(contrib[idx] > 0)  # direction toward predicted class

            # Numeric
            if raw_name.startswith("num__"):
                base_col = raw_name.split("__", 1)[1]
                raw_val = input_df.iloc[0].get(base_col, None)
                title = _humanize_numeric(base_col, raw_val)
                reasons.append(f"{title} {_direction_phrase(inc, predicted_label)}")
                continue

            # Categorical (mention only the active category)
            if x[idx] <= 0.5:
                continue
            nice = _prettify_feat_name(raw_name)
            label = _humanize_onehot(nice)
            if label is None:
                # Skip categories like Area-* to avoid clunky text
                continue
            reasons.append(f"{label} {_direction_phrase(inc, predicted_label)}")

        # De-duplicate
        seen, clean = set(), []
        for r in reasons:
            if r not in seen:
                clean.append(r); seen.add(r)

        if not clean:
            clean = ["The model combined several subtle signals; no single factor dominated."]

        return clean

    except Exception:
        return ["The model combined several signals for this prediction."]

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

    # --- Why this prediction? (chips + friendly bullet list)
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

        reasons = explain_with_importances(input_df, pre, clf, str(pred), top_k=3)
        # Render as bullets for readability
        st.markdown("<ul class='reason-list'>" + "".join([f"<li>{r}</li>" for r in reasons]) + "</ul>", unsafe_allow_html=True)

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
