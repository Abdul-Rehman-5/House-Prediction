# =========================================
# Production ML Script: House Price Model
# =========================================

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("house_price_regression_dataset.csv")

# Features & Target
X = df.drop("House_Price", axis=1)
y = df["House_Price"]

# ==============================
# 2. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 3. Define Models
# ==============================
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1)
}

best_model = None
best_rmse = float("inf")
best_name = ""

# ==============================
# 4. Cross-Validation for Model Selection
# ==============================
print("Cross-Validation Results:\n")

for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=5
    )

    rmse = -cv_scores.mean()
    print(f"{name} CV RMSE: {rmse:.2f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = pipeline
        best_name = name

# ==============================
# 5. Train Best Model on Full Training Data
# ==============================
best_model.fit(X_train, y_train)

# ==============================
# 6. Predictions
# ==============================
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# ==============================
# 7. Evaluation Metrics Function
# ==============================
def evaluate(y_true, y_pred, dataset_name="Dataset"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name} Metrics:")
    print(f"R2 Score: {r2*100:.2f}%")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

# Train vs Test Performance
evaluate(y_train, y_train_pred, "Train")
evaluate(y_test, y_test_pred, "Test")

# ==============================
# 8. Feature Importance (Coefficients)
# ==============================
if best_name in ["LinearRegression", "Ridge", "Lasso"]:
    model_coeffs = best_model.named_steps["model"].coef_

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model_coeffs
    }).sort_values(by="Coefficient", ascending=False)

    print("\nFeature Impact on House Price:")
    print(feature_importance)

# ==============================
# 9. Save Best Model
# ==============================
joblib.dump(best_model, "house_price_model.pkl")

print(f"\nBest Model Selected: {best_name}")
print(f"Best CV RMSE: {best_rmse:.2f}")
print("Model saved as: house_price_model.pkl")

# ==============================
# 10. Example Production Prediction
# ==============================
sample_input = pd.DataFrame([{
    "Square_Footage": 2500,
    "Num_Bedrooms": 3,
    "Num_Bathrooms": 2,
    "Year_Built": 2015,
    "Lot_Size": 2.5,
    "Garage_Size": 1,
    "Neighborhood_Quality": 7
}])

predicted_price = best_model.predict(sample_input)

print(f"\nSample Predicted House Price: {predicted_price[0]:.2f}")