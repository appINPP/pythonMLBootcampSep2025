import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

#--- Load dataset
df = sns.load_dataset("titanic")

print("Head of dataset:\n", df.head())
print("\nTail of dataset:\n", df.tail())
print("\nInfo:\n")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

#--- Data exploration
df['survived'].value_counts().plot(kind='bar', title="Class Distribution (0=dead, 1=survived)")
plt.show()

# Numerical distributions
df[['age','fare']].hist(bins=20, figsize=(10,4))
plt.suptitle("Feature Distributions")
plt.show()

#--- Preprocessing
# Drop columns with too many missing values or irrelevant info
df = df.drop(columns=['deck', 'embark_town', 'alive'])

# Fill missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Encode categorical variables
df = pd.get_dummies(df, columns=['sex','class','embarked','who','adult_male','alone'], drop_first=True)

# Features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#--- Build DNN model
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#--- Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=16,
    verbose=1
)

#--- Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

#--- Save weights + architecture
model.save_weights("titanic.weights.h5")

#--- Plot training history
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

