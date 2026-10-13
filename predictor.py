"""Prediction logic, kept separate from the web UI so it can be tested on its own."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODES = ("early", "later")
MAX_GRADE = 20.0


def load_model(mode: str) -> dict:
    """Load a saved bundle: {"pipeline": ..., "features": [...], "metrics": {...}}."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    path = MODEL_DIR / f"student_{mode}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 06_train_and_save.py first.")
    return joblib.load(path)


def predict_grade(values: dict, mode: str, bundle: dict | None = None) -> float:
    """Estimate the final grade (G3, 0-20) for one student.

    values: {column_name: value} for every feature the model needs.
    Extra keys are ignored, so the early model safely ignores G1/G2.
    """
    bundle = bundle or load_model(mode)
    missing = [c for c in bundle["features"] if c not in values]
    if missing:
        raise ValueError(f"Missing inputs: {missing}")
    X_new = pd.DataFrame([values])[bundle["features"]]  # same columns, same order as training
    raw = float(bundle["pipeline"].predict(X_new)[0])
    return float(np.clip(raw, 0, MAX_GRADE))