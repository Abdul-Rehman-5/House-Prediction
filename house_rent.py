import numpy as np
import pandas as pd
import joblib

import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, 
    r2_score,mean_absolute_percentage_error,
    mean_squared_error)



df = pd.read_csv("house_price_regression_dataset.csv")

#check data
# print(df.head())
# print(df.info())
# print(df.isnull().sum())

#No need for data handling
#No missing and categorical value
X = df.drop("House_Price", axis=1,)
y = df["House_Price"]

X_train,X_test,y_train,y_test, = train_test_split(
    X, y, random_state=42, test_size=0.2 )

scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # Diagonal line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Price')
plt.show()

# Instead of accuracy_score, use these:
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2 = r2_score(y_test, y_pred)

print(f"R2 Score: {r2*100:.2f}%")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

residuals = y_test - y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# ==============================
# 8. Feature Importance (Coefficients)
# ==============================
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nFeature Impact on Price:")
print(coefficients)

# ==============================
# Save Model & Scaler
# ==============================
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and scaler saved successfully.")



