import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)


# Load dataset again to preprocess new data the same way
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Scale features with the same scaler (in practice, you’d save the scaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Rebuild the same model architecture
model = keras.Sequential([
    keras.Input(shape=(X_scaled.shape[1],)),
    layers.Dense(30, activation='relu'),
    layers.Dense(15, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Load saved weights
model.load_weights("breast_cancer_weights.weights.h5")

# Make predictions
y_pred_proba = model.predict(X_scaled).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

#------ Evaluate the predictions:
print("Classification Report:\n")
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0, 1],
            yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "r--")  # diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


