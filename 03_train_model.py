
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# 1. Load the dataset
script_dir = Path(__file__).resolve().parent
csv_path = script_dir.parent / "data" / "student_performance.csv"

if not csv_path.exists():
    csv_path = script_dir / "data" / "student_performance.csv"

df = pd.read_csv(csv_path)

# 2. Separate features and target
X = df.drop(columns=["G3"])
y = df["G3"]

# 3. Identify numerical and categorical columns
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

# 4. Prepare the data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# 6. Build and train the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = root_mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- MODEL RESULTS ---")
print(f"MAE:  {mae:.2f} grade points")
print(f"RMSE: {rmse:.2f} grade points")
print(f"R²:   {r2:.3f}")

# 9. Compare actual and predicted grades
results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": predictions.round(2)
})

print("\n--- SAMPLE PREDICTIONS ---")
print(results.head(10).to_string(index=False))