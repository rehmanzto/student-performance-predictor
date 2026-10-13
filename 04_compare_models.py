
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# 1. Load data
script_dir = Path(__file__).resolve().parent
csv_path = script_dir.parent / "data" / "student_performance.csv"

if not csv_path.exists():
    csv_path = script_dir / "data" / "student_performance.csv"

df = pd.read_csv(csv_path)

print("COLUMNS IN DATASET:")
print(df.columns.tolist())

print("\nG1 exists:", "G1" in df.columns)
print("G2 exists:", "G2" in df.columns)


# Define target
target = "G3"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# Remove previous grades only if they exist
columns_to_exclude = [
    col for col in ["G1", "G2", "G3"]
    if col in df.columns
]

X = df.drop(columns=columns_to_exclude)
y = df[target]

print("Target:", target)
print("Excluded columns:", columns_to_exclude)
print("Number of features:", X.shape[1])
print("Features:", X.columns.tolist())

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

# 3. Split once: all models use the same test rows
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

# 4. Define the models
models = {
    "Baseline": DummyRegressor(strategy="mean"),

    "Linear Regression": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=42
        ))
    ])
}

# 5. Train and evaluate
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, predictions), 3),
        "RMSE": round(root_mean_squared_error(y_test, predictions), 3),
        "R2": round(r2_score(y_test, predictions), 3)
    })

results_df = pd.DataFrame(results)

print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))

print("\nLower MAE and RMSE are better.")
print("Higher R2 is generally better.")
