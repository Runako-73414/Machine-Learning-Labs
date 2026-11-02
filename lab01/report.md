Lab 01 Report

Goal
The goal of this lab was to train and evaluate baseline machine learning models using the Breast Cancer Wisconsin dataset.

Dataset
Breast Cancer Wisconsin dataset from scikit-learn.

Method
Loaded the dataset from scikit-learn.
Split the data into 70% training and 30% testing.
Trained two machine learning models:
Logistic Regression
Random Forest
Evaluated model performance using accuracy.
Measured model size, training time, inference latency, and memory usage.

Results
Accuracy

Model	Accuracy
Logistic Regression	0.941520
Random Forest	0.935673

Model Size
Model	Size (bytes)
Logistic Regression	1055
Random Forest	290889

Training Time
Model	Time (seconds)
Logistic Regression	0.3448
Random Forest	0.1412

Inference Latency
Model	Latency (seconds)
Logistic Regression	0.0002223
Random Forest	0.0021424

Memory Usage
Memory used during execution: 140,443,648 bytes.

Deployment Suitability
Platform	Logistic Regression	Random Forest
Cloud	Yes       	Yes
Edge	Yes	        Yes
Mobile	Yes	        Yes
TinyML	Yes	        No

Conclusions
Logistic Regression achieved the highest accuracy of 94.15%.
Logistic Regression produced a much smaller model (1,055 bytes) than Random Forest (290,889 bytes).
Random Forest trained faster than Logistic Regression.
Logistic Regression had lower inference latency.
Logistic Regression is suitable for TinyML deployment, while Random Forest is better suited for Cloud, Edge, and Mobile platforms.