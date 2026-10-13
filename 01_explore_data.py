
from pathlib import Path
from ucimlrepo import fetch_ucirepo

# Project folder location
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

# Step 1: Download/load the dataset
student_data = fetch_ucirepo(id=320)

# Step 2: Separate features and targets

X = student_data.data.features
grades = student_data.data.targets

df = X.copy()

# Add all three grade columns
for col in ["G1", "G2", "G3"]:
    if col in grades.columns:
        df[col] = grades[col].to_numpy()

DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_DIR / "student_performance.csv", index=False)

print("Dataset saved successfully!")
print("FINAL DATA SHAPE:", df.shape)
print("GRADE COLUMNS:", [ c for c in ["G1", "G2", "G3"] if c in df.columns])

# Step 3: Understand the data
print("FEATURE SHAPE:", X.shape)
print("TARGET COLUMNS:", grades.columns.tolist())

print("\nFIRST 5 FEATURE ROWS:")
print(X.head())

print("\nFIRST 5 TARGET ROWS:")
print(grades.head())

print("\nMISSING VALUES IN FEATURES:")
print(X.isna().sum().sum())

# Step 4: Save features + final grade
if "G3" not in grades.columns:
    raise ValueError("Could not find the G3 target column.")

df = X.copy()
df["G3"] = grades
print("\nDataset saved successfully!")
print("File:", DATA_DIR / "student_performance.csv")
print("FINAL DATA SHAPE:", df.shape)