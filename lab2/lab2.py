
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


print("STEP 1: Preparing the dataset")

digits = load_digits()
X = digits.data / 16.0
y_raw = digits.target

num_classes = 10
m = len(y_raw)

Y = np.zeros((m, num_classes))
Y[np.arange(m), y_raw] = 1.0

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.30,
    random_state=42,
    stratify=y_raw
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Input shape:", X.shape)
print("Output shape:", Y.shape)


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def compute_loss(Y_pred, Y_true):
    eps = 1e-12
    Y_pred = np.clip(Y_pred, eps, 1.0)
    return -np.sum(Y_true * np.log(Y_pred)) / len(Y_true)


def accuracy(Y_pred, Y_true):
    predictions = np.argmax(Y_pred, axis=1)
    actual = np.argmax(Y_true, axis=1)
    return np.mean(predictions == actual) * 100

class NeuralNetwork:

    def __init__(
        self,
        input_size=64,
        hidden_size=16,
        output_size=10,
        seed=42
    ):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(
            0, 0.01, size=(input_size, hidden_size)
        )
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = rng.normal(
            0, 0.01, size=(hidden_size, hidden_size)
        )
        self.b2 = np.zeros((1, hidden_size))

        self.W3 = rng.normal(
            0, 0.01, size=(hidden_size, output_size)
        )
        self.b3 = np.zeros((1, output_size))

    def forward(self, X):

        # First hidden layer
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)

        # Second hidden layer
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)

        # Output layer
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = softmax(self.Z3)

        return self.A3

    def backward_gradients(self, X, Y_true):

        batch_size = X.shape[0]

        # Output layer
        dZ3 = self.A3 - Y_true
        dW3 = self.A2.T @ dZ3 / batch_size
        db3 = np.sum(dZ3, axis=0, keepdims=True) / batch_size

        # Second hidden layer
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(self.Z2)
        dW2 = self.A1.T @ dZ2 / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size

        # First hidden layer
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = X.T @ dZ1 / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3
        }

    def update_parameters(self, gradients, learning_rate):

        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            parameter = getattr(self, name)
            parameter -= learning_rate * gradients[name]


print("\nSTEP 4: Checking the untrained network")

test_network = NeuralNetwork(hidden_size=16, seed=42)

sample_X = X_train[:32]
sample_Y = Y_train[:32]

sample_predictions = test_network.forward(sample_X)

print("Output shape:", sample_predictions.shape)

assert sample_predictions.shape == (32, 10)

row_sums = np.sum(sample_predictions, axis=1)

assert np.allclose(row_sums, 1.0, atol=1e-6)

initial_loss = compute_loss(sample_predictions, sample_Y)

print(f"Initial cross-entropy loss: {initial_loss:.6f}")
print(f"Expected approximate loss: {np.log(10):.6f}")

print("Output shape and softmax checks passed.")


def gradient_check(network, X_batch, Y_batch, num_checks=10):

    print("\nSTEP 5: Numerical gradient checking")

    # Calculate analytical gradients
    predictions = network.forward(X_batch)
    analytical = network.backward_gradients(X_batch, Y_batch)

    rng = np.random.default_rng(123)

    parameter_names = ["W1", "W2", "W3"]
    passed = 0
    epsilon = 1e-5

    for check in range(num_checks):

        # Randomly select a weight from the three weight matrices
        name = rng.choice(parameter_names)
        parameter = getattr(network, name)

        index = tuple(
            rng.integers(0, dimension)
            for dimension in parameter.shape
        )

        original_value = parameter[index]

        # Loss with weight + epsilon
        parameter[index] = original_value + epsilon
        loss_plus = compute_loss(
            network.forward(X_batch), Y_batch
        )

        # Loss with weight - epsilon
        parameter[index] = original_value - epsilon
        loss_minus = compute_loss(
            network.forward(X_batch), Y_batch
        )

        # Restore weight
        parameter[index] = original_value

        numerical_gradient = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        analytical_gradient = analytical[name][index]

        # Relative error
        denominator = max(
            1e-12,
            abs(numerical_gradient) + abs(analytical_gradient)
        )

        relative_error = (
            abs(numerical_gradient - analytical_gradient)
            / denominator
        )

        if relative_error < 1e-6:
            passed += 1

        print(
            f"Check {check + 1:2d}: {name}{index}, "
            f"analytical={analytical_gradient:.8e}, "
            f"numerical={numerical_gradient:.8e}, "
            f"relative error={relative_error:.3e}"
        )

    print(f"\nGradient checks passed: {passed}/{num_checks}")

    if passed >= 9:
        print("GRADIENT CHECK PASSED")
    else:
        print("GRADIENT CHECK DID NOT MEET THE REQUIRED THRESHOLD")

    return passed


# Use a small batch for gradient checking
gradient_network = NeuralNetwork(hidden_size=16, seed=7)

gradient_check(
    gradient_network,
    X_train[:8],
    Y_train[:8],
    num_checks=10
)


def train_network(
    hidden_size,
    learning_rate,
    X_train,
    Y_train,
    epochs=50,
    batch_size=32,
    seed=42
):

    network = NeuralNetwork(
        hidden_size=hidden_size,
        seed=seed
    )

    rng = np.random.default_rng(seed)

    loss_history = []
    train_accuracy_history = []

    start_time = time.perf_counter()

    for epoch in range(epochs):

        indices = rng.permutation(len(X_train))
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        for start in range(0, len(X_train), batch_size):

            end = min(start + batch_size, len(X_train))

            X_batch = X_shuffled[start:end]
            Y_batch = Y_shuffled[start:end]

            network.forward(X_batch)


            gradients = network.backward_gradients(
                X_batch, Y_batch
            )
            network.update_parameters(
                gradients, learning_rate
            )

        train_predictions = network.forward(X_train)
        epoch_loss = compute_loss(train_predictions, Y_train)
        epoch_accuracy = accuracy(train_predictions, Y_train)

        loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_accuracy)

        print(
            f"W={hidden_size:2d} | LR={learning_rate:.3f} | "
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss={epoch_loss:.6f} | "
            f"Train Accuracy={epoch_accuracy:.2f}%"
        )

    training_time = time.perf_counter() - start_time

    test_predictions = network.forward(X_test)
    test_loss = compute_loss(test_predictions, Y_test)
    test_accuracy = accuracy(test_predictions, Y_test)

    return {
        "network": network,
        "loss": loss_history,
        "train_accuracy": train_accuracy_history,
        "training_time": training_time,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }

hidden_widths = [16, 64]
learning_rates = [0.001, 0.01, 0.1]

results = {}

print("\nSTEP 6: Training all six configurations")

for width in hidden_widths:
    for lr in learning_rates:

        print("\n" + "=" * 65)
        print(f"Training network: W={width}, learning rate={lr}")
        print("=" * 65)

        result = train_network(
            hidden_size=width,
            learning_rate=lr,
            X_train=X_train,
            Y_train=Y_train,
            epochs=50,
            batch_size=32,
            seed=42
        )

        results[(width, lr)] = result


os.makedirs("results", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, width in zip(axes, hidden_widths):

    for lr in learning_rates:

        result = results[(width, lr)]

        ax.plot(
            range(1, 51),
            result["loss"],
            label=f"LR = {lr}"
        )

    ax.set_title(f"Hidden Width W = {width}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.grid(True)
    ax.legend()

plt.tight_layout()

output_path = os.path.join("results", "loss_curves.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"\nLoss curves saved to: {output_path}")

plt.show()


print("\nSTEP 8: Final test results")

print("-" * 85)
print(
    f"{'Width':<10}{'Learning Rate':<18}"
    f"{'Test Loss':<18}{'Test Accuracy':<20}{'Training Time (s)'}"
)
print("-" * 85)

for (width, lr), result in results.items():

    print(
        f"{width:<10}{lr:<18.3f}"
        f"{result['test_loss']:<18.6f}"
        f"{result['test_accuracy']:<20.2f}"
        f"{result['training_time']:.2f}"
    )


print("\nSTEP 9: Forward and backward timing")
print("Measuring median time over 100 runs.")

timing_network = NeuralNetwork(hidden_size=64, seed=99)

X_batch = X_train[:32]
Y_batch = Y_train[:32]

for _ in range(5):
    timing_network.forward(X_batch)
    timing_network.backward_gradients(X_batch, Y_batch)

forward_times = []
backward_times = []

for _ in range(100):

    start = time.perf_counter()
    timing_network.forward(X_batch)
    forward_times.append(time.perf_counter() - start)

    timing_network.forward(X_batch)
    start = time.perf_counter()
    timing_network.backward_gradients(X_batch, Y_batch)
    backward_times.append(time.perf_counter() - start)

median_forward = np.median(forward_times) * 1000
median_backward = np.median(backward_times) * 1000

print(f"Median forward time:  {median_forward:.6f} ms")
print(f"Median backward time: {median_backward:.6f} ms")

if median_forward > 0:
    print(
        "Backward/forward time ratio: "
        f"{median_backward / median_forward:.2f}x"
    )

print("\nLab 2 execution completed.")
