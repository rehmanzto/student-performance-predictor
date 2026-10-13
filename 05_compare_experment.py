"""Phase 6: compare early prediction (A) vs later prediction (B) of G3.

Experiment A: 30 student features         -> G3   (no G1/G2, true early prediction)
Experiment B: 30 student features + G1+G2 -> G3   (extra info, NOT a fair early-prediction comparison)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "student_performance.csv"
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV

df = pd.read_csv(csv_path)
assert {"G1", "G2", "G3"} <= set(df.columns), "Run 01_explore_data.py first: G1/G2/G3 missing"
print("Loaded:", df.shape)

y = df["G3"]
experiments = {
    "A: features only (early)": df.drop(columns=["G1", "G2", "G3"]),
    "B: features + G1 + G2 (later)": df.drop(columns=["G3"]),
}


def build(model, X):
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return Pipeline([("pre", pre), ("model", model)])


models = {
    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}

rows = []
for exp_name, X in experiments.items():
    for model_name, model in models.items():
        res = cross_validate(build(model, X), X, y, cv=cv, scoring=scoring)
        rows.append({
            "Experiment": exp_name,
            "Model": model_name,
            "R2": res["test_r2"].mean(),
            "RMSE": -res["test_rmse"].mean(),
            "MAE": -res["test_mae"].mean(),
        })

results = pd.DataFrame(rows)
pd.set_option("display.width", 120)
print("\n", results.round(3).to_string(index=False))

out = csv_path.parent / "experiment_results.csv"
results.to_csv(out, index=False)
print("\nSaved:", out)

# Which inputs matter most in B? (shows why G1/G2 dominate)
XB = experiments["B: features + G1 + G2 (later)"]
rf = build(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), XB).fit(XB, y)
names = rf.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 8 features in Experiment B:")
print(imp.head(8).round(3).to_string())