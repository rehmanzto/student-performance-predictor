
import importlib
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

try:
    sns = importlib.import_module("seaborn")
except ModuleNotFoundError:
    sns = None

# Find the CSV file
script_dir = Path(__file__).resolve().parent
possible_paths = [
    script_dir / "data" / "student_performance.csv",
    script_dir.parent / "data" / "student_performance.csv",
]

csv_path = next((p for p in possible_paths if p.exists()), None)

if csv_path is None:
    raise FileNotFoundError(
        "CSV not found. Check the data folder location."
    )

# Load the dataset
df = pd.read_csv(csv_path)

# 1. Understand the dataset
print("\n--- SHAPE ---")
print(df.shape)

print("\n--- FIRST 5 ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

# 2. Check missing values
print("\n--- MISSING VALUES ---")
print(df.isna().sum().sort_values(ascending=False).head(10))

# 3. Understand numerical data
print("\n--- SUMMARY STATISTICS ---")
print(df.describe().round(2))

# 4. Understand final grades
print("\n--- FINAL GRADE (G3) ---")
print(df["G3"].describe())
print("\nGrade counts:")
print(df["G3"].value_counts().sort_index())

# 5. Explore relationships with G3
columns_to_check = [
    c for c in ["studytime", "absences", "failures", "G1", "G2", "G3"]
    if c in df.columns
]

print("\n--- CORRELATION WITH G3 ---")
print(
    df[columns_to_check]
    .corr(numeric_only=True)["G3"]
    .sort_values(ascending=False)
)

# 6. Visualize grade distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["G3"], bins=20, kde=True)
plt.title("Distribution of Final Grades (G3)")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# 7. Compare study time and final grade
if "studytime" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="studytime", y="G3")
    plt.title("Study Time vs Final Grade")
    plt.xlabel("Weekly Study Time Category")
    plt.ylabel("Final Grade")
    plt.tight_layout()
    plt.show()