# train_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error,
    explained_variance_score
)

# Load and clean data
data = fetch_california_housing(as_frame=True)
df = data.frame.dropna()
print(df.head())

# Feature engineering
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
feature_names = X.columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)

# Evaluation
residuals = y_train - y_train_pred
print("\nTraining Evaluation Metrics:")
print(f"MAE: {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"MSE: {mean_squared_error(y_train, y_train_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_train, y_train_pred):.4f}")
print(f"R² Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"Explained Variance: {explained_variance_score(y_train, y_train_pred):.4f}")

# Save model and scaler
joblib.dump(model, "gradient_boosting_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ------------------
# Plots
# ------------------

# 1. Actual vs Predicted
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.4)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--r')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Training)")
plt.grid(True)
plt.tight_layout()
plt.savefig("train_actual_vs_predicted.png")
plt.show()

# 2. Residuals Histogram
plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=50, kde=True, color="steelblue")
plt.title("Residuals Distribution (Training)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("train_residuals_histogram.png")
plt.show()

# 3. Colored Scatter by Residuals
plt.figure(figsize=(7, 6))
scatter = plt.scatter(y_train, y_train_pred, c=residuals, cmap='coolwarm', alpha=0.6)
plt.colorbar(scatter, label="Residual")
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--k')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Colored by Residuals)")
plt.tight_layout()
plt.savefig("train_scatter_colored_residuals.png")
plt.show()

# 4. Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=feature_names[indices])
plt.title("Feature Importance (Gradient Boosting)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("train_feature_importance.png")
plt.show()

