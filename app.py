import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime as dt
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================
# Page & theme
# ==============================
st.set_page_config(page_title="Ride Cancellation Predictor", layout="centered")
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
# Model loader
# ==============================
@st.cache_resource
def load_model():
    return joblib.load("decision_tree_balanced.pkl")  # trained Pipeline(preprocessor + classifier)

model = load_model()

# Optional: load frozen priors/frequencies if you exported them from training
pickup_priors_path = Path("pickup_priors.csv")    # columns: Pickup Location,pickup_cancel_rate
drop_priors_path   = Path("drop_priors.csv")      # columns: Drop Location,drop_cancel_rate
pair_freqs_path    = Path("pair_freqs.csv")       # columns: Pickup Location,Drop Location,pickup_drop_pair_freq

pickup_priors = {}
drop_priors   = {}
pair_freqs    = {}

if pickup_priors_path.exists():
    dfp = pd.read_csv(pickup_priors_path)
    pickup_priors = dict(zip(dfp["Pickup Location"], dfp["pickup_cancel_rate"]))

if drop_priors_path.exists():
    dfd = pd.read_csv(drop_priors_path)
    drop_priors = dict(zip(dfd["Drop Location"], dfd["drop_cancel_rate"]))

if pair_freqs_path.exists():
    dff = pd.read_csv(pair_freqs_path)
    pair_freqs = {(r["Pickup Location"], r["Drop Location"]): r["pickup_drop_pair_freq"] for _, r in dff.iterrows()}

# ==============================
# Helpers
# ==============================
AREAS = [f"Area-{i}" for i in range(1, 51)]
VEHICLES = ["Auto", "Mini", "Sedan", "Bike"]
PAYMENTS = ["Cash", "Card"]
DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def derive_time_features(ts: dt.datetime):
    hour = ts.hour
    dow = ts.weekday()  # Monday=0
    weekend = 1 if dow >= 5 else 0
    if 5 <= hour <= 11: band = "Morning"
    elif 12 <= hour <= 16: band = "Afternoon"
    elif 17 <= hour <= 21: band = "Evening"
    else: band = "Night"
    return {"hour_of_day": hour, "day_of_week": dow, "is_weekend": weekend, "time_band": band, "day_name": DAY_NAMES[dow]}

def get_pair_freq(pickup, drop):
    if (pickup, drop) in pair_freqs:
        return float(pair_freqs[(pickup, drop)])
    return 50.0 if pickup == drop else 10.0  # fallback heuristic

def get_pickup_prior(pickup, time_band):
    if pickup in pickup_priors:
        return float(pickup_priors[pickup])
    return 0.25 if time_band == "Night" else 0.10  # fallback heuristic

def get_drop_prior(drop, time_band):
    if drop in drop_priors:
        return float(drop_priors[drop])
    return 0.20 if time_band == "Night" else 0.08  # fallback heuristic

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

# ==============================
# Session State
# ==============================
if "ui_stage" not in st.session_state:
    st.session_state.ui_stage = "landing"  # landing -> inputs -> predicted
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
st.markdown('<div class="subtitle">Simple. Clear. Explainable.</div>', unsafe_allow_html=True)

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
        dcol, tcol, pcol = st.columns([1,1,1])
        with dcol:
            booking_date = st.date_input("Date", value=dt.date.today())
        with tcol:
            booking_time = st.time_input("Time", value=dt.datetime.now().time())
        with pcol:
            preset = st.selectbox("Quick set", ["None","Now","Rush hour (08:30)","Late night (23:30)"], index=0)
            if preset == "Now":
                now = dt.datetime.now()
                booking_date, booking_time = now.date(), now.time()
            elif preset == "Rush hour (08:30)":
                booking_time = dt.time(8, 30)
            elif preset == "Late night (23:30)":
                booking_time = dt.time(23, 30)

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
            # Predict
            pred = model.predict(input_df)[0]
            try:
                proba = model.predict_proba(input_df)[0]
            except Exception:
                proba = None

            # Store in session
            st.session_state.last_input_df = input_df
            st.session_state.last_time_feats = time_feats
            st.session_state.last_pred = pred
            st.session_state.last_proba = proba
            st.session_state.ui_stage = "predicted"
            st.experimental_rerun()

# ==============================
# Predicted view: output then explain
# ==============================
if st.session_state.ui_stage == "predicted":
    input_df = st.session_state.last_input_df
    time_feats = st.session_state.last_time_feats
    pred = st.session_state.last_pred
    proba = st.session_state.last_proba

    # --- Prediction banner
    st.markdown("### 🔮 Prediction")
    pred_lower = str(pred).lower()
    if "success" in pred_lower:
        st.success(f"✅ Predicted Booking Status: **{pred}**")
    elif "cancel" in pred_lower:
        st.error(f"❌ Predicted Booking Status: **{pred}**")
    else:
        st.warning(f"⚠️ Predicted Booking Status: **{pred}**")

    # --- Short "Why this?" summary (lightweight, no heavy libs)
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
        st.markdown('<div class="muted">Detailed visuals are available in the tabs below—generated only if you click their buttons.</div>', unsafe_allow_html=True)

    st.divider()

    # ==============================
    # Tabs for lazy-loaded visuals
    # ==============================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Class probability",
        "🪄 Explain this prediction",
        "🧩 Confusion matrix",
        "🏁 Global feature importance",
        "🔀 What-if analysis",
    ])

    # --- Tab 1: Class probability (on click)
    with tab1:
        st.markdown("**Why this class, not others?** See the probability for each class.")
        if st.button("Show class probabilities"):
            if proba is None:
                st.info("Model does not expose probabilities.")
            else:
                prob_df = pd.DataFrame({"Class": model.classes_, "Probability": proba})
                st.bar_chart(prob_df.set_index("Class"))
                top = model.classes_[int(np.argmax(proba))]
                st.caption(f"Top class: **{top}** with {100*float(np.max(proba)):.1f}% confidence.")

    # --- Tab 2: Explain this prediction (SHAP if available)
    with tab2:
        st.markdown("**Which features pushed this prediction?**")
        clicked = st.button("Explain with SHAP (if available)")
        if clicked:
            try:
                import shap
                pre = model.named_steps["preprocessor"]
                clf = model.named_steps["classifier"]
                X_trans = pre.transform(input_df)
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_trans)

                # Convert to a simple bar of top |SHAP| features for the predicted class
                cls_index = list(model.classes_).index(pred)
                vals = np.abs(shap_values[cls_index][0])
                names = pre.get_feature_names_out()
                order = np.argsort(vals)[::-1][:15]
                top_names = names[order]
                top_vals = vals[order]

                fig, ax = plt.subplots()
                ax.barh(range(len(top_vals)), top_vals)
                ax.set_yticks(range(len(top_vals)))
                ax.set_yticklabels(top_names)
                ax.invert_yaxis()
                ax.set_xlabel("|Contribution|")
                ax.set_title("Top feature contributions (approx.)")
                st.pyplot(fig)
                st.caption("Higher bars indicate features with stronger influence toward the predicted class (approximate).")
            except Exception as e:
                st.info(f"SHAP-based explanation not available: {e}")

    # --- Tab 3: Confusion matrix (only if you provide validation files)
    with tab3:
        st.markdown("**How reliable is the model overall?**")
        st.write("Provide small validation files to render this:")
        st.code("val_X_sample.csv, val_y_sample.csv", language="text")
        if st.button("Show confusion matrix from validation sample"):
            x_path, y_path = Path("val_X_sample.csv"), Path("val_y_sample.csv")
            if x_path.exists() and y_path.exists():
                Xv = pd.read_csv(x_path)
                yv = pd.read_csv(y_path).iloc[:, 0]
                ypred = model.predict(Xv)
                cm = confusion_matrix(yv, ypred, labels=model.classes_)
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(ax=ax, cmap="Blues", colorbar=False)
                ax.set_title("Confusion Matrix (sample)")
                st.pyplot(fig)
            else:
                st.info("Upload val_X_sample.csv and val_y_sample.csv to show this.")

    # --- Tab 4: Global feature importance (tree-based)
    with tab4:
        st.markdown("**What matters most overall?**")
        if st.button("Show global feature importance"):
            try:
                pre = model.named_steps["preprocessor"]
                clf = model.named_steps["classifier"]
                if hasattr(clf, "feature_importances_"):
                    names = pre.get_feature_names_out()
                    importances = clf.feature_importances_
                    order = np.argsort(importances)[::-1][:20]
                    fig, ax = plt.subplots()
                    ax.barh(range(len(order)), importances[order])
                    ax.set_yticks(range(len(order)))
                    ax.set_yticklabels(names[order])
                    ax.invert_yaxis()
                    ax.set_title("Top global feature importances")
                    ax.set_xlabel("Importance")
                    st.pyplot(fig)
                else:
                    st.info("This classifier does not expose feature_importances_.")
            except Exception as e:
                st.info(f"Could not compute importances: {e}")

    # --- Tab 5: What-if analysis (simple per-feature sweep)
    with tab5:
        st.markdown("**What if we change one feature?**")
        st.write("Pick a numeric feature and see how class probabilities move when we vary it around the current value.")
        numeric_candidates = ["hour_of_day","pickup_cancel_rate","drop_cancel_rate","pickup_drop_pair_freq"]
        feat = st.selectbox("Feature to vary", numeric_candidates, index=0)
        span = st.slider("± range around current value", 1, 24, 6)
        if st.button("Run what-if"):
            base = input_df.iloc[0].to_dict()
            center = float(base[feat])
            # build sweep
            if feat == "hour_of_day":
                grid = list(range(int(max(0, center - span)), int(min(23, center + span)) + 1))
            else:
                # numeric continuous sweep (20 points)
                lo = max(0.0, center - span * 0.05)
                hi = min(1.0, center + span * 0.05) if "rate" in feat else max(center + span, center + span)  # simple bound
                grid = np.linspace(lo, hi, 20)

            rows = []
            for v in grid:
                row = base.copy()
                row[feat] = float(v)
                df_try = pd.DataFrame([row])
                try:
                    probs = model.predict_proba(df_try)[0]
                    rows.append([v, *probs])
                except Exception:
                    # fallback to class only
                    pred_try = model.predict(df_try)[0]
                    probs_vec = [np.nan]*len(model.classes_)
                    idx = list(model.classes_).index(pred_try)
                    probs_vec[idx] = 1.0
                    rows.append([v, *probs_vec])

            plot_df = pd.DataFrame(rows, columns=[feat, *model.classes_]).set_index(feat)
            st.line_chart(plot_df)
            st.caption("Trend lines show how class probabilities respond to the chosen feature (approximate, ceteris paribus).")

    st.divider()
    # Back / New prediction CTA
    cols = st.columns(2)
    if cols[0].button("← New prediction", use_container_width=True):
        st.session_state.ui_stage = "inputs"
        st.experimental_rerun()
    if cols[1].button("🏠 Back to start", use_container_width=True):
        st.session_state.ui_stage = "landing"
        st.experimental_rerun()
