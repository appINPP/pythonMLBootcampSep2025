# predict_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error,
    explained_variance_score
)

# Load data
data = fetch_california_housing(as_frame=True)
df = data.frame.dropna()

# Load model and scaler
model = joblib.load("gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

# Prepare features
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)
residuals = y - y_pred

# Evaluation
print("\nPrediction Evaluation Metrics (Full Dataset):")
print(f"MAE: {mean_absolute_error(y, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y, y_pred):.4f}")
print(f"R² Score: {r2_score(y, y_pred):.4f}")
print(f"Explained Variance: {explained_variance_score(y, y_pred):.4f}")

# ------------------
# Plots
# ------------------

# 1. Actual vs Predicted
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y, y=y_pred, alpha=0.4)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Full Data)")
plt.grid(True)
plt.tight_layout()
plt.savefig("predict_actual_vs_predicted.png")
plt.show()

# 2. Residuals Histogram
plt.figure(figsize=(8, 4))
sns.histplot(residuals, bins=50, kde=True, color="darkorange")
plt.title("Residuals Distribution (Full Data)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("predict_residuals_histogram.png")
plt.show()

# 3. Colored Scatter by Residuals
plt.figure(figsize=(7, 6))
scatter = plt.scatter(y, y_pred, c=residuals, cmap='coolwarm', alpha=0.6)
plt.colorbar(scatter, label="Residual")
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--k')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Colored by Residuals)")
plt.tight_layout()
plt.savefig("predict_scatter_colored_residuals.png")
plt.show()

