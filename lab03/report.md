Lab 03: Architectures and Their Costs

Goal

The objective of this lab was to implement and compare different neural network architectures for image classification using the Fashion-MNIST dataset. The architectures were evaluated based on classification accuracy, parameter count, computational cost (FLOPs), and inference latency. The aim was to determine which architecture provided the best balance between performance and computational cost.

Dataset

The Fashion-MNIST dataset was used for all experiments.

Dataset characteristics:

Training samples: 60,000
Testing samples: 10,000
Image size: 28 × 28 pixels
Number of classes: 10
Batch size: 64
Training batches: 938
Testing batches: 157

Images were converted to tensors using transforms.ToTensor() and loaded through PyTorch DataLoaders.

Architecture 1: MLP
Structure
Flatten
Linear (784 → 128)
ReLU
Linear (128 → 10)
Results

Training loss:

Epoch 1: 0.5456
Epoch 2: 0.4024
Epoch 3: 0.3627
Epoch 4: 0.3348
Epoch 5: 0.3177

Test Accuracy:

86.38%

Estimated FLOPs:

203,264

Discussion

The MLP provided good performance while maintaining a simple architecture and low computational cost.

Architecture 2: Deep MLP
Structure
Flatten
Linear (784 → 256)
ReLU
Linear (256 → 128)
ReLU
Linear (128 → 64)
ReLU
Linear (64 → 10)
Results

Training loss:

Epoch 1: 0.5505
Epoch 2: 0.3817
Epoch 3: 0.3390
Epoch 4: 0.3157
Epoch 5: 0.2955

Test Accuracy:

86.44%

Estimated FLOPs:

484,608

Discussion

The Deep MLP slightly improved accuracy over the standard MLP but required substantially more computations.

Architecture 3: CNN
Structure
Conv2D (1 → 16)
ReLU
MaxPool2D
Conv2D (16 → 32)
ReLU
MaxPool2D
Flatten
Linear (1568 → 128)
ReLU
Linear (128 → 10)
Results

Training loss:

Epoch 1: 0.4967
Epoch 2: 0.3255
Epoch 3: 0.2768
Epoch 4: 0.2453
Epoch 5: 0.2242

Test Accuracy:

90.56%

Estimated FLOPs:

~1.2 Million

Discussion

The CNN achieved the highest classification accuracy among all architectures. Convolutional layers effectively captured spatial features and patterns present in image data.

Architecture 4: Attention Model
Structure
Split image into 16 patches
Linear patch projection (49 → 64)
Transformer Encoder
Mean Pooling
Linear classifier (64 → 10)
Results

Training loss:

Epoch 1: 0.7889
Epoch 2: 0.5151
Epoch 3: 0.4446
Epoch 4: 0.4049
Epoch 5: 0.3800

Test Accuracy:

84.69%

Parameter Count

Manual parameter count:

285,002 parameters

FLOPs

Estimated FLOPs:

~1.5 Million

THOP Profiling Results

Measured results:

THOP FLOPs: 50,816
THOP Parameters: 3,850
Comparison Between Estimated and Measured Values

A significant difference was observed between the manual calculations and the THOP profiling results. This is likely because the TransformerEncoder implementation is not fully supported by THOP and some internal attention operations are not counted correctly during profiling. Therefore, the manually calculated FLOP estimate provides a more realistic estimate of the model's actual computational cost.

Inference Latency

Latency was measured using repeated forward passes of the Attention Model.

Batch Size 1

0.009680 seconds

Batch Size 128

0.015028 seconds

Discussion

Increasing the batch size from 1 to 128 resulted in only a small increase in latency, demonstrating efficient batch processing.

Architecture Comparison

Final test accuracies:

MLP: 86.38%
Deep MLP: 86.44%
CNN: 90.56%
Attention Model: 84.69%

Accuracy ranking:

CNN (90.56%)
Deep MLP (86.44%)
MLP (86.38%)
Attention Model (84.69%)
Computational Cost Comparison

Estimated FLOPs:

MLP: 203,264
Deep MLP: 484,608
CNN: ~1.2 Million
Attention Model: ~1.5 Million

The Attention Model had the highest computational complexity, while the standard MLP had the lowest.

Trade-Off Analysis
Accuracy vs FLOPs

The CNN achieved the highest accuracy while maintaining lower computational requirements than the Attention Model.

The Deep MLP required more computations than the standard MLP but provided only a small improvement in classification accuracy.

The Attention Model had the highest estimated computational cost but did not achieve superior classification performance.

Accuracy vs Parameters

The Attention Model contained considerably more parameters than the other architectures but did not produce the highest accuracy.

The CNN demonstrated more efficient use of model parameters and achieved the strongest classification performance.

Accuracy vs Latency

Although the Attention Model demonstrated relatively low inference latency, its classification accuracy remained below that of the CNN.

The CNN provided the most favorable balance between accuracy and computational requirements.

Most Cost-Effective Architecture

The CNN was determined to be the most cost-effective architecture because:

It achieved the highest test accuracy (90.56%).
It effectively captured spatial image features.
It provided better performance than the Attention Model despite lower computational complexity.
It offered the best balance between accuracy and computational cost.
Generated Figures

The following figures were generated and saved in the results folder:

attention_loss_curve.png
accuracy_comparison.png

The loss curve illustrates the convergence behavior of the Attention Model during training.

The accuracy comparison figure visualizes the classification performance of the MLP, Deep MLP, CNN, and Attention architectures.

Conclusions

Four neural network architectures were implemented and evaluated on the Fashion-MNIST dataset.

Final test accuracies were:

MLP: 86.38%
Deep MLP: 86.44%
CNN: 90.56%
Attention Model: 84.69%

The CNN achieved the highest test accuracy and provided the strongest balance between performance and computational cost.

The Deep MLP only slightly improved performance compared to the standard MLP despite requiring additional computations.

The Attention Model demonstrated the use of self-attention mechanisms for image classification, but it did not outperform the CNN on this dataset.

Overall, the CNN was the most effective and cost-efficient architecture for Fashion-MNIST classification.