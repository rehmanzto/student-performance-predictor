"""Analytics page: how good are the models, and what is related to grades in this dataset?

Streamlit shows any file in the pages/ folder as an extra page next to app.py.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"
DATA_PATH = BASE / "data" / "student_performance.csv"
if not DATA_PATH.exists():
    DATA_PATH = Path(r"C:\Users\user\Desktop\data\student_performance.csv")

DROP = {"early": ["G1", "G2", "G3"], "later": ["G3"]}

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def test_split(mode):
    """Recreate the exact 20% test set used in 06_train_and_save.py (same seed, same split)."""
    df = load_data()
    X = df.drop(columns=DROP[mode])
    _, X_test, _, y_test = train_test_split(X, df["G3"], test_size=0.2, random_state=42)
    return X_test, y_test


@st.cache_data
def test_predictions(mode):
    bundle = joblib.load(MODEL_DIR / f"student_{mode}.joblib")
    X_test, y_test = test_split(mode)
    pred = np.clip(bundle["pipeline"].predict(X_test), 0, 20)
    table = pd.DataFrame({"Actual G3": y_test.to_numpy(), "Predicted G3": pred})
    return table, bundle["metrics"]


@st.cache_data
def feature_importance(mode):
    """Permutation importance: shuffle one column at a time and see how much R² drops."""
    bundle = joblib.load(MODEL_DIR / f"student_{mode}.joblib")
    X_test, y_test = test_split(mode)
    result = permutation_importance(
        bundle["pipeline"], X_test, y_test, n_repeats=10, random_state=42, scoring="r2"
    )
    out = pd.DataFrame({"Feature": X_test.columns, "Importance": result.importances_mean})
    return out.sort_values("Importance", ascending=False)


if not (MODEL_DIR / "student_early.joblib").exists():
    st.error("Model files not found. Run `python 06_train_and_save.py` first.")
    st.stop()

df = load_data()
st.title("📊 Analytics")
st.caption("Based on 649 students from two Portuguese schools. Patterns here are not proof of cause and effect.")

# ---------- 1. the data ----------
st.header("1. The data")
c1, c2, c3 = st.columns(3)
c1.metric("Students", len(df))
c2.metric("Average final grade (G3)", f"{df['G3'].mean():.1f} / 20")
c3.metric("Share with G3 ≥ 10 (pass)", f"{(df['G3'] >= 10).mean():.0%}")

dist = df["G3"].value_counts().reindex(range(21), fill_value=0)
st.bar_chart(dist.rename("Students"), x_label="Final grade (G3)", y_label="Number of students")

# ---------- 2. the models ----------
st.header("2. How good are the models?")
st.caption("Scores come from the 130 students the models never saw during training.")
tab_early, tab_later = st.tabs(["Early model (no grades)", "Later model (with G1 + G2)"])
for tab, mode in ((tab_early, "early"), (tab_later, "later")):
    with tab:
        preds, metrics = test_predictions(mode)
        a, b, c = st.columns(3)
        a.metric("MAE (average error)", f"{metrics['mae']:.2f}")
        b.metric("RMSE", f"{metrics['rmse']:.2f}")
        c.metric("R²", f"{metrics['r2']:.2f}")
        st.scatter_chart(preds, x="Actual G3", y="Predicted G3")
        st.caption("Each dot is one student. The closer the dots are to a diagonal line, the better the model.")

# ---------- 3. what is related to grades ----------
st.header("3. What matters before any grades exist?")
imp = feature_importance("early").head(10)
st.bar_chart(imp, x="Feature", y="Importance", sort="-Importance", horizontal=True)
st.caption(
    "Importance = how much the early model's R² drops when that column is shuffled. "
    "Small values and a small test set mean the ranking is rough. "
    "It shows what the model relies on, not what causes good grades."
)

st.subheader("Explore one feature")
feature = st.selectbox("Feature", [c for c in df.columns if c not in ("G1", "G2", "G3")])
series = df[feature]
if pd.api.types.is_numeric_dtype(series) and series.nunique() > 12:
    groups = pd.cut(series, bins=5).astype(str)
else:
    groups = series
avg = df.groupby(groups, observed=True)["G3"].agg(["mean", "count"]).reset_index()
avg.columns = [feature, "Average G3", "Students"]
st.bar_chart(avg, x=feature, y="Average G3")
st.dataframe(avg, hide_index=True)
st.caption("Check the Students column: averages over very few students are unreliable.")