import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print("data.target_names: ",data.target_names)

#----- Explore and Prepare the Dataset: 
#Explore dataset
print("Head of dataset:\n", df.head(), " target/label: ",df['target'])
print("\nTail of dataset:\n", df.tail())
print("\nInfo:\n")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

#Basic plots
df['target'].value_counts().plot(kind='bar', title="Class Distribution")
plt.show()

df.hist(bins=20, figsize=(12, 10))
plt.suptitle("Feature Distributions")
plt.show()

#Preprocessing
# Remove nulls if any (this dataset has none, but we’ll demonstrate)
df = df.dropna()

X = df.drop('target', axis=1)
y = df['target']

#Scale features/variables for DNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#--------- The DNN part:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Define the model
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(30, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(15, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

#Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=1
)

#Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

#Save model weights
model.save_weights("breast_cancer_weights.weights.h5")

# ---- Plot training history
plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

