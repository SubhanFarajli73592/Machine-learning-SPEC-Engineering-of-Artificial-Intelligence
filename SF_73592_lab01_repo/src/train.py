import numpy as np
import random
import csv
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Lock random numbers so the test results are identical every time
random.seed(42)
np.random.seed(42)

# Load the case files and split: 70% study material, 30% sealed exam
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

# Train Detective 1 (The Scout) and grade their test
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_acc = lr.score(X_test, y_test)

# Train Detective 2 (The 100-person Council) and grade their test
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test, y_test)

# Write their final test scores onto a report card file
with open("lab01/results/baseline_accuracy.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Accuracy"])
    writer.writerow(["LogisticRegression", f"{lr_acc:.4f}"])
    writer.writerow(["RandomForestClassifier", f"{rf_acc:.4f}"])

print("Success! Models trained and accuracy saved.")