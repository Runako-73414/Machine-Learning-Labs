from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# Train models
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

# Accuracy
lr_acc = accuracy_score(y_test, lr.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# Save results
results = pd.DataFrame({
    "model": ["LogisticRegression", "RandomForest"],
    "accuracy": [lr_acc, rf_acc]
})

results.to_csv(
    "lab01/results/baseline_accuracy.csv",
    index=False
)

print(results)
print("Results saved successfully!")

import joblib
import os

# Save models
joblib.dump(lr, "lab01/results/logistic_model.joblib")
joblib.dump(rf, "lab01/results/randomforest_model.joblib")

# Model sizes
lr_size = os.path.getsize("lab01/results/logistic_model.joblib")
rf_size = os.path.getsize("lab01/results/randomforest_model.joblib")

print("Logistic Model Size:", lr_size, "bytes")
print("Random Forest Size:", rf_size, "bytes")

import time

start = time.perf_counter()
lr.fit(X_train, y_train)
lr_time = time.perf_counter() - start

start = time.perf_counter()
rf.fit(X_train, y_train)
rf_time = time.perf_counter() - start

print("Logistic Regression Training Time:", lr_time)
print("Random Forest Training Time:", rf_time)

start = time.perf_counter()
lr.predict(X_test[:1])
lr_latency = time.perf_counter() - start

start = time.perf_counter()
rf.predict(X_test[:1])
rf_latency = time.perf_counter() - start

print("Logistic Regression Latency:", lr_latency)
print("Random Forest Latency:", rf_latency)

import psutil
import os

process = psutil.Process(os.getpid())

print("Memory Usage:", process.memory_info().rss, "bytes")