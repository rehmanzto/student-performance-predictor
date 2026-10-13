"""Rebuild student_performance.csv with all three grade columns (G1, G2, G3).

Run:  python .\\01_fix_dataset.py
Needs: pip install ucimlrepo pandas
"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(r"C:\Users\user\Desktop\data")
CSV_PATH = DATA_DIR / "student_performance.csv"

student_data = fetch_ucirepo(id=320)  # UCI "Student Performance" (649 rows)

X = student_data.data.features
grades = student_data.data.targets

print("Features:", X.shape)
print("Target columns found:", list(grades.columns))

df = X.copy()
for col in ["G1", "G2", "G3"]:
    if col in grades.columns:
        df[col] = grades[col].to_numpy()
    else:
        print(f"WARNING: {col} not found in targets")

DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(CSV_PATH, index=False)

print("\nSaved to:", CSV_PATH)
print("FINAL DATA SHAPE:", df.shape)
print("GRADE COLUMNS:", [c for c in ["G1", "G2", "G3"] if c in df.columns])