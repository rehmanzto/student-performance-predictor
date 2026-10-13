"""Tests for predictor.py. They need models/ and data/ to exist (run 06_train_and_save.py first)."""
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

from predictor import MAX_GRADE, MODES, load_model, predict_grade

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "student_performance.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def bundles():
    return {mode: load_model(mode) for mode in MODES}


def student(df, i=0):
    """One real student as a plain dict, like the web form produces."""
    return df.iloc[i].to_dict()


# ---------- loading ----------
@pytest.mark.parametrize("mode", MODES)
def test_bundle_has_expected_parts(bundles: dict[str, dict], mode: tuple[Literal['early'], Literal['later']]):
    assert set(bundles[mode]) >= {"pipeline", "features", "metrics"}


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        load_model("sometimes")


# ---------- which columns each model uses ----------
def test_early_model_does_not_use_grades(bundles: dict[str, dict]):
    features = bundles["early"]["features"]
    assert not {"G1", "G2", "G3"} & set(features)


def test_later_model_uses_g1_g2_but_never_the_target(bundles: dict[str, dict]):
    features = bundles["later"]["features"]
    assert {"G1", "G2"} <= set(features)
    assert "G3" not in features


# ---------- predictions ----------
@pytest.mark.parametrize("mode", MODES)
def test_predictions_stay_inside_grade_range(df: pd.DataFrame, bundles: dict[str, dict], mode: tuple[Literal['early'], Literal['later']]):
    for i in range(0, 60):
        p = predict_grade(student(df, i), mode, bundles[mode])
        assert 0 <= p <= MAX_GRADE


def test_later_model_reacts_to_g2(df: pd.DataFrame, bundles: dict[str, dict]):
    low = {**student(df), "G1": 4, "G2": 4}
    high = {**student(df), "G1": 18, "G2": 18}
    assert predict_grade(high, "later", bundles["later"]) > predict_grade(low, "later", bundles["later"])


def test_early_model_ignores_g1_g2(df: pd.DataFrame, bundles: dict[str, dict]):
    base = student(df)
    changed = {**base, "G1": 0, "G2": 0}
    assert predict_grade(base, "early", bundles["early"]) == predict_grade(changed, "early", bundles["early"])


def test_missing_input_raises(df, bundles):
    values = student(df)
    values.pop(bundles["early"]["features"][0])
    with pytest.raises(ValueError):
        predict_grade(values, "early", bundles["early"])


# ---------- sanity check on quality ----------
def test_later_model_is_better_than_early(bundles):
    early, later = bundles["early"]["metrics"], bundles["later"]["metrics"]
    assert later["r2"] > early["r2"]
    assert later["mae"] < early["mae"]
    assert later["r2"] > 0.5