"""Student Performance Predictor - Streamlit app (Phase 10).

Run:  streamlit run app.py

Needs:
  models/student_early.joblib and models/student_later.joblib   (from 06_train_and_save.py)
  data/student_performance.csv   (only used to build the form: value ranges and options)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
DATA_PATH = BASE / "data" / "student_performance.csv"
if not DATA_PATH.exists():  # fall back to the original location
    DATA_PATH = Path(r"C:\Users\user\Desktop\data\student_performance.csv")

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")


# ---------- loading (cached, so Streamlit doesn't reload on every click) ----------
@st.cache_resource
def load_bundle(mode: str):
    return joblib.load(MODEL_DIR / f"student_{mode}.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def make_input(col: str, series: pd.Series, container):
    """Build the right widget for a column, using the data for ranges/options."""
    if pd.api.types.is_numeric_dtype(series):
        return container.number_input(
            col,
            min_value=int(series.min()),
            max_value=int(series.max()),
            value=int(series.median()),
            step=1,
        )
    options = sorted(series.dropna().unique().tolist())
    default = series.mode().iloc[0]
    return container.selectbox(col, options, index=options.index(default))


# ---------- page ----------
st.title("🎓 Student Performance Predictor")
st.caption(
    "A learning project. Trained on 649 students from two Portuguese schools, so it is "
    "not validated for Bangladeshi HSC students. Treat results as a rough estimate."
)

if not (MODEL_DIR / "student_early.joblib").exists():
    st.error("Model files not found. Run `python 06_train_and_save.py` first.")
    st.stop()

df = load_data()

mode_label = st.radio(
    "Which situation is this?",
    ["Early estimate (no grades yet)", "Later estimate (G1 and G2 are known)"],
    horizontal=True,
)
mode = "early" if mode_label.startswith("Early") else "later"

bundle = load_bundle(mode)
features = bundle["features"]
metrics = bundle["metrics"]

with st.sidebar:
    st.header("Model quality")
    st.caption(f"Measured on {metrics['n_test']} students the model never saw.")
    st.metric("MAE (average error)", f"{metrics['mae']:.2f} points")
    st.metric("RMSE", f"{metrics['rmse']:.2f} points")
    st.metric("R²", f"{metrics['r2']:.2f}")
    if mode == "early":
        st.warning("Without earlier grades the model is only a rough guide.")

st.subheader("Student information")
with st.form("student_form"):
    values = {}
    columns = st.columns(3)
    for i, col in enumerate(features):
        values[col] = make_input(col, df[col], columns[i % 3])
    submitted = st.form_submit_button("Estimate final grade", type="primary")

if submitted:
    X_new = pd.DataFrame([values])[features]  # same columns, same order as training
    estimate = float(np.clip(bundle["pipeline"].predict(X_new)[0], 0, 20))
    typical_error = metrics["mae"]
    low, high = max(0.0, estimate - typical_error), min(20.0, estimate + typical_error)

    st.subheader("Result")
    c1, c2 = st.columns(2)
    c1.metric("Estimated final grade (G3)", f"{estimate:.1f} / 20")
    c2.metric("Typical error", f"± {typical_error:.1f} points")
    st.progress(estimate / 20)
    st.write(f"Rough range: **{low:.1f} to {high:.1f}** (estimate ± typical error).")
    st.caption("In Portuguese schools, 10 out of 20 is the pass mark.")

    if estimate < 10:
        st.info("The estimate is below the pass mark. Early support could help.")
    else:
        st.success("The estimate is at or above the pass mark.")