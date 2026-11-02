FINAL LAB 05 REPORT

INTRODUCTION

The objective of this lab was to implement a scalar automatic differentiation engine, verify gradient correctness, compare results against PyTorch autograd and numerical differentiation, and investigate execution behavior in PyTorch. The lab also explored computation graph construction, reverse-mode automatic differentiation, execution optimizations, and tensor memory layouts.


VALUE NODE DESIGN

A custom Value class was implemented to represent nodes in a computation graph.

Each node stores:

• Scalar value (data)
• Gradient (grad)
• Parent nodes (_prev)
• Operation metadata (_op)
• Local backward function (_backward)

This structure allows computation graphs to be built dynamically during the forward pass and traversed during gradient computation.


SUPPORTED OPERATIONS

The following operations were implemented:

Addition

    a + b

Multiplication

    a * b

Power

    a ** n

ReLU

    a.relu()

Each operation creates a new Value object and stores the parent nodes together with the local derivative rule required during backpropagation.


BACKWARD PASS IMPLEMENTATION

The backward pass was implemented using reverse-mode automatic differentiation.

The process consisted of:

1. Building a topological ordering of the computation graph.
2. Setting the output node gradient to 1.
3. Traversing the graph in reverse topological order.
4. Executing local backward functions at each node.
5. Accumulating gradients using addition rather than overwriting previous values.

This ensures correct gradient propagation through arbitrary computation graphs.


GRADIENT VERIFICATION

The custom engine was evaluated using:

    ((a * b + c) ** 2).relu()

with:

    a = 2
    b = 3
    c = 1

CUSTOM AUTOGRAD RESULTS

    Output = 49.0

    a.grad = 42.0
    b.grad = 28.0
    c.grad = 14.0

PYTORCH AUTOGRAD RESULTS

    a.grad = 42.0
    b.grad = 28.0
    c.grad = 14.0

GRADIENT DIFFERENCES

    Difference a = 0.0
    Difference b = 0.0
    Difference c = 0.0

NUMERICAL GRADIENT CHECK

    Numerical Gradient = 41.99999999698889

    Autograd Gradient = 42.0

    Difference = 3.011109583894722e-09

The difference was significantly smaller than the required threshold:

    1e-6

Therefore, the custom automatic differentiation engine successfully matched both PyTorch autograd and numerical differentiation.


MAKE_MOONS EXPERIMENT

A neural network was trained on the make_moons dataset.

ARCHITECTURE

    Input: 2
    Hidden: 8
    Output: 1

ACTIVATION FUNCTIONS

    ReLU
    Sigmoid

LOSS FUNCTION

    Binary Cross Entropy

OPTIMIZER

    Adam

RESULTS

    Test Accuracy = 0.9267

    Median Training Time = 0.3684 seconds

The network successfully learned the nonlinear decision boundary of the dataset.

The PyTorch implementation achieved a median training time of 0.3684 seconds across five runs and served as the framework baseline for comparison.


FRAMEWORK INVESTIGATION

TENSOR INSPECTION

A sample tensor was examined.

Results:

    Shape: torch.Size([10, 2])

    Dtype: torch.float32

    Contiguous: True


EAGER VS TRACED EXECUTION

Runtime was evaluated using standard eager execution and a traced model created with torch.jit.trace().

EAGER EXECUTION

    Median Runtime = 0.0749 seconds

TRACED EXECUTION

    Median Runtime = 0.0616 seconds

DISCUSSION

The traced model executed faster than the eager version.

Tracing reduced Python overhead and allowed the computation graph to be optimized for repeated execution.

Measured improvement:

    Approximately 17.7%


NON-CONTIGUOUS TENSOR PERFORMANCE

A transposed tensor was used to create a non-contiguous memory layout.

RESULTS

    Contiguous Tensor:
    0.0411 seconds

    Non-Contiguous Tensor:
    0.0619 seconds

DISCUSSION

The non-contiguous tensor required approximately 50.7% more execution time than the contiguous tensor.

This performance difference occurs because non-contiguous tensors use less efficient memory access patterns and may require additional indexing operations.


WHEN COMPILATION HELPS

Compilation or tracing is most beneficial when the same computation graph is executed repeatedly.

Benefits include:

• Reduced Python interpreter overhead
• Optimized graph execution
• Faster repeated inference
• Improved deployment performance

The benefits become more significant for larger models and workloads that repeatedly execute the same graph structure.


CONCLUSION

A scalar automatic differentiation engine was successfully implemented from scratch.

The engine correctly constructed computation graphs, performed reverse-mode automatic differentiation, accumulated gradients, and matched both PyTorch autograd and numerical differentiation.

A neural network trained on the make_moons dataset achieved a test accuracy of 92.67%.

Framework-level investigations showed that traced execution outperformed eager execution and that non-contiguous tensors incurred measurable performance overhead.

Overall, the lab demonstrated the principles underlying modern deep learning frameworks and highlighted the impact of execution mode and memory layout on performance.