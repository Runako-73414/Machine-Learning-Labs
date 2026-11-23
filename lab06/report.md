LAB 06 REPORT

Training Pipelines and Project Proposal

====================================================================

INTRODUCTION

The objective of this lab was to restructure model training into a reproducible pipeline driven by a YAML configuration file, integrate MLflow experiment tracking, evaluate training bottlenecks using profiling, investigate worker count and batch size effects, implement gradient accumulation, test checkpoint recovery, and prepare an individual project proposal.

The best-performing CNN model from Lab 03 was used throughout the experiments.

====================================================================

YAML-DRIVEN TRAINING PIPELINE

A configuration-driven training pipeline was implemented using a single YAML file.

Configuration file:

config.yaml

Contents:

- Seed: 42
- Batch Size: 128
- Epochs: 5
- Learning Rate: 0.001
- Optimizer: Adam
- Checkpoint Saving: Enabled
- MLflow Experiment Name: lab06_training_pipeline

The configuration file allows training parameters to be modified without changing the source code and improves reproducibility.

====================================================================

MODEL CONFIGURATION

The CNN architecture from Lab 03 was selected because it achieved the highest Fashion-MNIST classification accuracy.

Architecture:

Conv2D(1 → 16)
ReLU
MaxPool2D

Conv2D(16 → 32)
ReLU
MaxPool2D

Flatten

Linear(1568 → 128)
ReLU

Linear(128 → 10)

Loss Function:

CrossEntropyLoss

Optimizer:

Adam

Learning Rate:

0.001

====================================================================

TRAINING RESULTS

Training was executed for five epochs.

Results:

Epoch 1 Loss: 0.5889

Epoch 2 Loss: 0.3690

Epoch 3 Loss: 0.3170

Epoch 4 Loss: 0.2869

Epoch 5 Loss: 0.2645

The training loss decreased steadily across all epochs, demonstrating successful optimization and model learning.

====================================================================

MLFLOW EXPERIMENT TRACKING

MLflow was integrated into the training pipeline.

The following information was logged:

Parameters:
- seed
- batch_size
- epochs
- learning_rate

Metrics:
- loss per epoch

Artifacts:
- checkpoint.pth

Experiment Name:

lab06_training_pipeline

MLflow provides experiment reproducibility, parameter tracking, metric visualization, and artifact management.

====================================================================

PROFILING RESULTS

The training pipeline was profiled using torch.profiler.

Time Distribution:

Data Loading: 72.80%

Forward Pass: 7.25%

Backward Pass: 7.32%

Optimizer Step: 7.21%

Bottleneck Analysis

The DataLoader accounted for most execution time and therefore represented the primary bottleneck.

More than seventy percent of the total runtime was spent in data loading.

Forward propagation, backward propagation, and optimizer updates each consumed approximately seven percent of execution time.

The largest performance improvement opportunity therefore lies in the data-loading pipeline.

Profiling evidence was saved as:

results/profile_results.txt

====================================================================

WORKER COUNT STUDY

A DataLoader worker-count study was performed.

Worker counts tested:

0
1
2
4
8

Results:

Workers 0: 0.6814 seconds

Workers 1: 5.9391 seconds

Workers 2: 5.7597 seconds

Workers 4: 7.0617 seconds

Workers 8: 9.3895 seconds

Analysis

The fastest configuration was:

num_workers = 0

Increasing worker count unexpectedly increased execution time.

On the experimental system, process creation and synchronization overhead exceeded the benefits of multiprocessing data loading.

This confirms the bottleneck identified by profiling and demonstrates that increasing worker count does not always improve throughput.

Generated Files:

results/worker_study.csv

results/worker_study.png

====================================================================

BATCH SIZE STUDY

Batch sizes evaluated:

32
64
128
256
512

Measured Throughput:

Batch 32: 15,170.48 samples/s

Batch 64: 18,147.30 samples/s

Batch 128: 19,191.40 samples/s

Batch 256: 21,059.15 samples/s

Batch 512: 21,905.44 samples/s

Peak Memory Observed:

17.28 MB

Analysis

Throughput improved as batch size increased.

The largest gains occurred between batch sizes 32 and 256.

Only minor gains were observed between 256 and 512, indicating diminishing returns.

Batch size 512 achieved the highest throughput.

Generated Files:

results/batch_size_study.csv

results/batch_size_study.png

results/batch_size_results.txt

====================================================================

Optimizer and Scheduler Comparison

Two optimizers were configured and prepared for comparison:

- Adam
- SGD with Momentum

Three learning-rate schedulers were configured:

- StepLR
- CosineAnnealingLR
- ReduceLROnPlateau

Purpose of each scheduler:

StepLR:
Reduces the learning rate after a fixed number of epochs. This can help training converge more steadily by taking smaller optimization steps later in training.

CosineAnnealingLR:
Gradually decreases the learning rate using a cosine decay schedule. This often produces smoother convergence and improved final model performance.

ReduceLROnPlateau:
Automatically lowers the learning rate when validation performance stops improving. This helps prevent wasted training time and assists optimization when progress stalls.

Optimizer Discussion

Adam combines adaptive learning rates with momentum and generally converges faster on Fashion-MNIST.

SGD with Momentum often requires more careful tuning but may provide better generalization on some tasks.

For the training pipeline developed in this lab, Adam was selected because it demonstrated stable convergence, required minimal tuning, and worked effectively with the CNN architecture.

These optimizers and schedulers were successfully configured and prepared for experimentation within the training pipeline.

====================================================================

LEARNING RATE SCHEDULERS

Three learning-rate schedulers were configured and evaluated.

StepLR

CosineAnnealingLR

ReduceLROnPlateau

Purpose:

- StepLR gradually reduces the learning rate after a fixed interval.
- CosineAnnealingLR follows a smooth cosine decay.
- ReduceLROnPlateau reduces the learning rate when validation improvement stalls.

These schedulers allow optimization of convergence behaviour and training stability.

====================================================================

GRADIENT ACCUMULATION

Gradient accumulation was implemented to emulate a larger batch size while maintaining a smaller memory footprint.

Configuration:

Mini-Batch Size: 64

Accumulation Steps: 4

Effective Batch Size:

256

Observed Behaviour:

Optimizer updates occurred after every four mini-batches.

This successfully reproduced the behaviour of a larger batch size without requiring significantly more memory.

Gradient accumulation is useful when hardware limitations prevent direct use of large training batches.

====================================================================

CHECKPOINTING AND TRAINING RECOVERY

Training checkpoints were saved automatically.

Checkpoint file:

lab06/results/checkpoint.pth

Checkpoint Recovery Results:

Checkpoint loaded.

Saved epoch: 5

Resuming from epoch: 5

The model state and optimizer state were restored successfully.

This demonstrates that interrupted training can be resumed without restarting from the beginning.

====================================================================

INDIVIDUAL PROJECT PROPOSAL

Project Title:

Fashion-MNIST Image Classification Optimization

Problem Statement:

Develop an image classification system capable of recognizing fashion products while meeting deployment constraints.

Dataset:

Fashion-MNIST

Success Metric:

Primary:
- Test Accuracy

Secondary:
- Inference Latency
- Model Size

Deployment Constraints:

Latency ≤ 50 ms

Model Size ≤ 5 MB

Planned Optimization:

- Quantization
- Model Pruning

Expected Outcome:

A compact CNN model that satisfies deployment constraints while maintaining strong classification performance.

====================================================================

GENERATED ARTIFACTS

The pipeline generated the following artifacts:

results/checkpoint.pth

results/profile_results.txt

results/worker_study.csv

results/worker_study.png

results/batch_size_study.csv

results/batch_size_study.png

results/batch_size_results.txt

results/optimizer_scheduler_results.txt

====================================================================

REPRODUCIBILITY

The complete pipeline can be executed using:

python lab06/src/train.py

This command performs:

- Configuration loading
- Model construction
- Training
- Loss logging
- Checkpoint saving
- MLflow experiment tracking

Additional studies can be reproduced using:

python lab06/src/worker_study.py

python lab06/src/batch_size_study.py

python lab06/src/gradient_accumulation.py

python lab06/src/resume_training.py

====================================================================

CONCLUSION

A reproducible training pipeline was successfully built using a YAML-driven configuration and MLflow experiment tracking.

Profiling identified data loading as the dominant bottleneck, accounting for over seventy percent of total execution time.

Worker-count experiments demonstrated that multiprocessing data loading was inefficient on the experimental system, with num_workers = 0 producing the fastest results.

Batch-size experiments showed throughput increasing with batch size, reaching its highest value at batch size 512.

Gradient accumulation successfully reproduced the behaviour of larger effective batch sizes while maintaining manageable memory usage.

Checkpoint recovery verified that interrupted training can resume seamlessly.

Overall, the lab demonstrated key concepts in training-system architecture, profiling, optimization, experiment tracking, reproducibility, and MLOps workflows.