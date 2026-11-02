Lab 02 Report

Goal

The goal of this lab was to build a neural network from scratch using NumPy and evaluate its performance on the Digits dataset. The network was implemented using forward propagation, backpropagation, mini-batch gradient descent, and gradient checking.

Dataset

The Digits dataset from scikit-learn was used for this experiment.

The dataset contains 1,797 handwritten digit samples. Each sample consists of 64 input features representing an 8×8 grayscale image of a handwritten digit. The dataset includes 10 output classes corresponding to the digits 0 through 9.

Data Preparation

The input features were normalized by dividing all pixel values by 16.0, scaling the data to the range [0, 1].

The class labels were converted into one-hot encoded vectors.

The dataset was split into:

- 70% training data (1,257 samples)
- 30% testing data (540 samples)

A stratified split with a random state of 42 was used to maintain class balance.

Neural Network Architecture

The neural network consisted of:

- Input layer: 64 neurons
- Hidden layer 1: 64 neurons
- Hidden layer 2: 64 neurons
- Output layer: 10 neurons

The ReLU activation function was used in the hidden layers, while the Softmax activation function was used in the output layer.

The network was trained using mini-batch gradient descent with a batch size of 32 and cross-entropy loss.

Initial Results

Before training, the network produced near-random predictions.

- Initial Loss: 2.302583
- Initial Accuracy: 12.01%

These results were expected because the weights were randomly initialized.

Training Configuration

The best-performing configuration was:

- Hidden Layer Width (W): 64
- Learning Rate: 0.1
- Batch Size: 32
- Epochs: 50

Training Results

Using the optimal configuration:

- Final Loss: 0.057780
- Final Training Accuracy: 96.82%

The training loss decreased significantly over 50 epochs, demonstrating successful learning.

Test Results

The trained network was evaluated on previously unseen test data.

- Test Accuracy: 94.63%

The high test accuracy indicates that the model generalized well and did not simply memorize the training set.

Loss Curve

The training loss was recorded over all 50 epochs and plotted using Matplotlib.

The loss curve was saved as:

`lab02/results/loss_curves.png`

The curve shows a consistent decrease in loss throughout training, confirming that the network converged successfully.

Timing Results

Forward and backward passes were executed 100 times, and the median execution time was recorded.

- Median Forward Pass Time: 0.001094 seconds
- Median Backward Pass Time: 0.001786 seconds

The backward pass required more time than the forward pass because gradient computation involves additional matrix operations and parameter updates.

Gradient Checking

Gradient checking was performed using the finite-difference method with ε = 1e-5.

A network weight was perturbed by a small positive and negative value, and the corresponding change in loss was measured to estimate the numerical gradient.

The numerical gradient was compared with the analytical gradient obtained through backpropagation. This procedure was used to verify the correctness of the gradient calculations and validate the implementation of backpropagation.

Conclusions

A neural network with two hidden layers was successfully implemented from scratch using NumPy.

The network was trained using mini-batch gradient descent, ReLU activation functions, Softmax output activation, and cross-entropy loss.

The best-performing model used:

- Hidden Layer Width: 64
- Learning Rate: 0.1
- Batch Size: 32
- Epochs: 50

The final training accuracy achieved was 96.82%, while the final test accuracy was 94.63%.

These results demonstrate that the neural network successfully learned to classify handwritten digits and achieved strong generalization performance on unseen data.