import time
import os
import psutil
import joblib
import numpy as np
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

random.seed(42)
np.random.seed(42)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
single_sample = X_test[0:1]

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

process = psutil.Process(os.getpid())

for name, model in models.items():
    print(f"--- {name} ---")
    
    # 1. Warm-up round so the computer wakes up
    model.fit(X_train, y_train) 
    
    # 2. Time the training speed across 5 repeat runs
    train_times = []
    for _ in range(5):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - start)
    
    train_mem_bytes = process.memory_info().rss
    
    # 3. Time 100 fast reaction tests on a single patient
    inf_times = []
    for _ in range(100):
        start = time.perf_counter()
        model.predict(single_sample)
        inf_times.append(time.perf_counter() - start)
        
    inf_mem_bytes = process.memory_info().rss
    
    # 4. Save the model to a backpack file and weigh it
    model_path = f"lab01/results/model_{name.replace(' ', '')}.joblib"
    joblib.dump(model, model_path)
    
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024
    
    # Print the physical stats
    print(f"Median Training Time: {np.median(train_times) * 1000:.2f} ms")
    print(f"Peak Training Memory: {train_mem_bytes / (1024*1024):.2f} MB")
    print(f"Median Inference Time: {np.median(inf_times) * 1000:.2f} ms")
    print(f"Peak Inference Memory: {inf_mem_bytes / (1024*1024):.2f} MB")
    print(f"Model Size: {size_bytes} bytes ({size_kb:.2f} KB)\n")