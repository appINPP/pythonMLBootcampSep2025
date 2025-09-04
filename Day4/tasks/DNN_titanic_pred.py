# titanic_prediction.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------
# 1. Load and preprocess dataset again
# --------------------------
df = sns.load_dataset("titanic")
df = df.drop(columns=['deck', 'embark_town', 'alive'])
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
df = pd.get_dummies(df, columns=['sex','class','embarked','who','adult_male','alone'], drop_first=True)

X = df.drop('survived', axis=1)
y = df['survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# 2. Rebuild model and load weights
# --------------------------
model = keras.Sequential([
    keras.Input(shape=(X_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.load_weights("titanic.weights.h5")

# --------------------------
# 3. Predictions
# --------------------------
y_pred_proba = model.predict(X_scaled).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# --------------------------
# 4. Evaluation
# --------------------------
print("Classification Report:\n")
print(classification_report(y, y_pred))  # 0/1 labels only

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[0, 1],
            yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (0=dead, 1=survived)")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "r--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


