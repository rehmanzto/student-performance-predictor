import joblib
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")
CSV_PATH = Path(r"C:\Users\user\Desktop\data\student_performance.csv")

bundle = joblib.load(MODEL_DIR / "student_later.joblib")
df = pd.read_csv(CSV_PATH)

# TODO 1: take one real student row, e.g. df.iloc[[5]]
# TODO 2: keep only the columns in bundle["features"]
# TODO 3: predict, clip to 0-20, print it
# TODO 4: change G1 and G2 by a few points and predict again. What happens?
# TODO 5: do the same with student_early.joblib. Does changing G1/G2 matter there?