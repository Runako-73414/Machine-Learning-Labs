Individual Project Proposal

Name: Runako Mapfungautsi

Student ID: 73414

Project Title

Fashion Image Classification Optimization

Problem Statement

The objective of this project is to design and optimize an image-classification system capable of accurately recognizing fashion products from grayscale Fashion-MNIST images while satisfying strict deployment constraints.

The project will focus on maintaining high predictive performance while reducing model size and inference latency for practical deployment scenarios.

Dataset

Dataset: Fashion-MNIST

Source: torchvision.datasets.FashionMNIST

Description:

Fashion-MNIST contains 70,000 grayscale images of clothing products belonging to 10 categories.

Training Samples: 60,000

Test Samples: 10,000

Image Size: 28 × 28 pixels

Classes: 10

Success Metrics

Primary Metric:

* Test Accuracy

Target:

* Greater than 90% classification accuracy

Secondary Metrics:

* Inference latency
* Model size
* Throughput

Deployment Constraints

The final model must satisfy:

* Latency ≤ 50 milliseconds
* Model Size ≤ 5 MB

These constraints are intended to simulate deployment on resource-constrained devices.

Planned Optimization Techniques

The project will evaluate:

* Model Pruning
   - Remove less important weights to reduce model complexity.

* Quantization
   - Convert model parameters to lower-precision formats to decrease model size and improve inference speed.

The impact of each optimization technique will be measured using accuracy, latency, and model-size comparisons.

Expected Outcome

The expected outcome a compressed CNN-based Fashion-MNIST classifier that:

- Achieves more than 90% test accuracy
- Meets the latency constraint of 50 milliseconds or less
- Remains smaller than 5 MB
- Demonstrates the effectiveness of pruning and quantization techniques

FEASIBILITY

The project is feasible because:

* Fashion-MNIST is publicly available and well documented.
* A high-performing CNN baseline has already been implemented in previous labs.
* PyTorch provides built-in support for model optimization techniques such as pruning and quantization.
* The project requirements can be completed within the available development timeframe.

CONCLUSION

This project aims to develop an efficient Fashion-MNIST image classification model optimized for deployment environments with strict latency and size constraints. By applying model pruning and quantization techniques, the project seeks to maintain high predictive performance while reducing computational and storage requirements.
