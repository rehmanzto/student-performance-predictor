"""Phase 8-9: train the final models, test them on held-out data, save them, try a prediction.

Two models:
  early -> the 30 student features only          (usable before any grade exists)
  later -> the 30 student features + G1 + G2     (usable once those grades are known)

Run:  python .\06_train_and_save.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = Path(r"C:\Users\user\Desktop\data\student_performance.csv")
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)
y = df["G3"]

# which columns to remove from the inputs for each mode
MODES = {
    "early": ["G1", "G2", "G3"],
    "later": ["G3"],
}


def build_pipeline(X):
    """Preprocessing + model in ONE object, so saving it saves both."""
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("model", model)])


for mode, drop_cols in MODES.items():
    X = df.drop(columns=drop_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline(X)
    pipe.fit(X_train, y_train)

    # grades live on a 0-20 scale, so keep predictions inside it
    pred = np.clip(pipe.predict(X_test), 0, 20)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    bundle = {"pipeline": pipe, "features": list(X.columns), "metrics": metrics}
    joblib.dump(bundle, MODEL_DIR / f"student_{mode}.joblib")
    (MODEL_DIR / f"metrics_{mode}.json").write_text(json.dumps(metrics, indent=2))

    print(f"[{mode}] test MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.3f}")

print("\nSaved models to:", MODEL_DIR)

# ---- smoke test: load the saved model the way the app will ----
bundle = joblib.load(MODEL_DIR / "student_early.joblib")
row = df.iloc[[0]]
X_new = row[bundle["features"]]          # same columns, same order as training
estimate = float(np.clip(bundle["pipeline"].predict(X_new)[0], 0, 20))
print(f"\nSmoke test (student 0): estimated G3 = {estimate:.1f}, actual G3 = {row['G3'].iloc[0]}")
print("(student 0 was in the training data, so this only proves load + predict works)")