# 🩺 Breast Cancer Prediction using KNN

# 1️⃣ Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2️⃣ Load Dataset

df = pd.read_csv('breast-cancer.csv')

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 3️⃣ Data Preprocessing

# Remove unnecessary column
df.drop(labels=['id'], axis=1, inplace=True)

# Splitting features and target
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# 4️⃣ Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5️⃣ Train KNN Model (k=3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# 6️⃣ Model Evaluation

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 7️⃣ Finding Best K Value

scores = []

for i in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

# Plot Accuracy vs K
plt.figure()
plt.plot(range(1, 16), scores)
plt.xlabel("Value of K")
plt.ylabel("Accuracy")
plt.title("K Value vs Accuracy")
plt.show()

# Print best K
best_k = scores.index(max(scores)) + 1
print("\nBest K value:", best_k)
print("Best Accuracy:", max(scores))

