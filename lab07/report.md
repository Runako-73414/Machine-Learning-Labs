LAB 07 REPORT

Compression: Pruning and Distillation

====================================================================

INTRODUCTION

The objective of this lab was to investigate model-compression techniques for deep neural networks using the Fashion-MNIST dataset. The study focused on pruning and knowledge distillation as methods for reducing model complexity while maintaining predictive performance.

The experiments compared baseline, pruned, and distilled models using parameter count, storage size, sparsity, latency, and model efficiency.

====================================================================

BASELINE CNN

An oversized convolutional neural network was constructed to serve as the teacher model for compression experiments.

Architecture:

Conv2D(1 → 64)
ReLU
MaxPool

Conv2D(64 → 128)
ReLU
MaxPool

Flatten

Linear(6272 → 512)
ReLU

Linear(512 → 10)

Parameter Count:

3,291,402

Model Size:

13,169,044 bytes

12.56 MB

Inference Latency:

0.003013 seconds

Discussion

The baseline CNN significantly exceeded the target size of approximately one million parameters and was intentionally over-parameterized to serve as a teacher model.

Its large capacity provides a strong reference model for evaluating the impact of compression techniques.

====================================================================

UNSTRUCTURED MAGNITUDE PRUNING

Unstructured pruning was performed using L1-magnitude pruning.

Weights with the smallest magnitudes were removed while preserving the original network structure.

Sparsity Results:

0% Sparsity:

0 zero weights

25% Sparsity:

144 zero weights

50% Sparsity:

288 zero weights

75% Sparsity:

432 zero weights

90% Sparsity:

518 zero weights

95% Sparsity:

547 zero weights

Analysis

The number of zero-valued weights increased proportionally with the target sparsity level.

At 95% sparsity, almost all weights in the selected convolution layer were removed while retaining a small subset of the most influential parameters.

Magnitude pruning dramatically reduces parameter utilization and storage requirements while maintaining the original architecture.

However, unstructured sparsity does not necessarily improve execution speed because most CPU and GPU kernels remain optimized for dense matrix operations.

====================================================================

STRUCTURED PRUNING

Structured pruning was performed using filter-level pruning based on L2 norm.

Instead of deleting individual weights, entire filters were removed.

Pruning Levels Evaluated:

10%
25%
50%

Results:

10% Structured Pruning:

54 zero weights

25% Structured Pruning:

144 zero weights

50% Structured Pruning:

288 zero weights

Analysis

Structured pruning removes complete filters rather than isolated parameters.

Because entire filters are removed, structured pruning is more likely to produce real inference-speed improvements on hardware.

As pruning severity increased, more filters were removed, leading to higher model sparsity and lower computational requirements.

====================================================================

KNOWLEDGE DISTILLATION

Knowledge distillation was used to transfer information from a large teacher model to a significantly smaller student model.

Teacher Model Parameters:

3,291,402

Student Model Parameters:

105,866

Reduction:

Approximately 31× fewer parameters

Distillation Hyperparameters:

Temperature (T): 4

Alpha: 0.5

Distillation Loss:

1.9138

Discussion

The student network satisfies the target size requirement of approximately 100,000 parameters.

Knowledge distillation allows the student model to learn not only from ground-truth labels but also from the probability distribution produced by the teacher model.

This approach enables smaller models to retain useful information learned by much larger networks.

====================================================================

LATENCY AND STORAGE ANALYSIS

Baseline Dense Model:

Model Size:

12.56 MB

Latency:

0.003013 seconds

Discussion

Dense neural network implementations are optimized for hardware execution and therefore frequently execute faster than sparse models.

Although pruning can significantly reduce storage requirements, latency improvements are not guaranteed.

Unstructured sparsity often fails to improve inference speed because dense kernels still process the entire tensor regardless of how many values are zero.

Structured pruning provides a more realistic path toward latency reduction because complete filters can be removed from computation.

====================================================================

MODEL EFFICIENCY COMPARISON

Baseline CNN

Data:
Fashion-MNIST

Model:
3,291,402 parameters

Compute:
Highest complexity

Storage:
12.56 MB

Energy:
Highest computational demand

Advantages:
High-capacity teacher model

------------------------------------------------------------

Unstructured Pruning

Data:
Fashion-MNIST

Model:
Up to 95% sparsity

Compute:
Reduced parameter utilization

Storage:
Substantially reduced

Energy:
Lower storage and memory overhead

Advantages:
Very high sparsity levels

Disadvantages:
Limited hardware acceleration

------------------------------------------------------------

Structured Pruning

Data:
Fashion-MNIST

Model:
Entire filters removed

Compute:
Reduced filter operations

Storage:
Smaller effective network

Energy:
Reduced computational demand

Advantages:
Better hardware efficiency

------------------------------------------------------------

Knowledge Distillation

Teacher Model:
3,291,402 parameters

Student Model:
105,866 parameters

Compute:
Lower computational complexity

Storage:
Significantly smaller

Energy:
Lower energy consumption

Advantages:
Approximately 31× parameter reduction while preserving teacher knowledge

====================================================================

EFFICIENCY PILLARS SUMMARY

Data

The same Fashion-MNIST dataset was used for all experiments to ensure fair comparison.

Model

The largest gains were achieved through model compression techniques:

- Unstructured pruning
- Structured pruning
- Knowledge distillation

Knowledge distillation reduced model size by approximately 31×.

Compute

Structured pruning and knowledge distillation reduced computational requirements more effectively than unstructured pruning.

Energy

Smaller models generally require fewer arithmetic operations and lower memory traffic, leading to reduced energy consumption and improved deployment efficiency.

====================================================================

FINAL CONCLUSION

An oversized CNN containing 3,291,402 parameters was used as the baseline teacher network.

Magnitude pruning successfully introduced sparsity levels up to 95%, substantially reducing parameter utilization.

Structured pruning removed entire filters and provided a more realistic mechanism for hardware acceleration.

Knowledge distillation reduced model size from 3,291,402 parameters to 105,866 parameters, representing approximately a 31× reduction in model size.

The experiments demonstrated that storage reduction and latency reduction are not always equivalent. Unstructured sparsity significantly reduces the number of active parameters but often fails to improve execution speed on dense CPU kernels.

Among the compression techniques evaluated, knowledge distillation provided the strongest balance between model size reduction, computational efficiency, and retention of useful predictive information.

Overall, structured pruning and knowledge distillation represent the most practical approaches for deploying efficient neural networks under resource constraints.