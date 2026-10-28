Lab 01 Report: Environment Setup and System Profiling




1. Objective
This experiment evaluates the computational demands of two distinct machine learning models trained on the Breast Cancer Wisconsin dataset within an isolated Python environment. By profiling training duration, single-sample latency, peak memory footprint, and serialized disk storage, we assess the deployment feasibility of each model across four hardware deployment tiers: Cloud, Edge, Mobile, and TinyML.



2. Methodology
Initialized a standalone Python 3.11 virtual environment (.venv) with the necessary dependencies.

Documented package specifications using print_versions.py and exported the configuration to results/versions.txt.

Partitioned the Breast Cancer Wisconsin dataset into a 70% training set and a 30% evaluation set using a fixed random seed of 42.

Trained two classification baselines: Logistic Regression and a 100-estimator Random Forest.

Logged final test classification accuracy to results/baseline_accuracy.csv.

Profiled computational metrics via measure.py:

Training duration: Measured across 5 iterations following an initial warm-up execution; the median duration was logged.

Single-instance inference latency: Evaluated by predicting a single sample over 100 trials, capturing the median latency.

Memory consumption: Monitored peak memory utilization during execution using the psutil library.

Storage footprint: Serialized trained artifacts to disk using joblib.dump and recorded final sizes in bytes and kilobytes.



3. Experimental Results
Baseline Accuracy

Model,Test Accuracy
Logistic Regression,0.9415
Random Forest Classifier,0.9357

Metric,Logistic Regression,Random Forest
Median Training Time,839.11 ms,358.27 ms
Median Inference Latency,0.15 ms,7.52 ms
Peak Inference Memory,103.86 MB,104.64 MB
Model Artifact Size,"1,055 bytes (1.03 KB)","290,889 bytes (284.07 KB)"

Platform Tier,Constraint Profile,Logistic Regression,Random Forest
Cloud,"RAM ≥ 1 GB, Latency ≤ 100 ms, Size ≤ 500 MB",Compatible,Compatible
Edge,"RAM 256–1024 MB, Latency ≤ 50 ms, Size ≤ 50 MB",Compatible,Compatible
Mobile,"RAM 64–256 MB, Latency ≤ 20 ms, Size ≤ 10 MB",Compatible,Compatible
TinyML,"RAM ≤ 256 KB, Latency ≤ 10 ms, Size ≤ 100 KB",Compatible,Incompatible (Exceeds size ceiling)

4. Key Takeaways
Model complexity does not guarantee superior performance: Logistic Regression outperformed the Random Forest model (0.9415 vs. 0.9357 test accuracy). For small, low-noise clinical datasets, a linear decision boundary generalizes well without overfitting to minor sample variance.

Structural simplicity significantly reduces inference latency: Logistic Regression generated predictions in 0.15 ms, compared to 7.52 ms for the ensemble model. Evaluating 100 hierarchical decision trees requires approximately 50 times more computation than evaluating a single linear dot product.

Artifact footprint dictates bare-metal edge feasibility: At 1.03 KB, the Logistic Regression parameters integrate seamlessly into memory-restricted microcontrollers. Conversely, the 284.07 KB Random Forest binary breaches the 100 KB TinyML threshold, restricting its practical deployment to mobile, edge, or cloud infrastructure.