import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time
import os

# Lock the dice roll so results match every single time
np.random.seed(42)

# -------------------------------------------------------------
# 1. Prepare the Letter Clues (Dataset)
# -------------------------------------------------------------
X, y = load_digits(return_X_y=True)
X = X / 16.0  # Scale ink brightness to range [0, 1]

# Set up the 10 mailboxes (one-hot labels: e.g. digit 3 -> mailbox 3)
num_classes = 10
Y_one_hot = np.zeros((y.shape[0], num_classes))
Y_one_hot[np.arange(y.shape[0]), y] = 1.0

# 70% study material, 30% sealed exam
X_train, X_test, y_train, y_test = train_test_split(
    X, Y_one_hot, test_size=0.30, stratify=y, random_state=42
)

# -------------------------------------------------------------
# 2. Workshop Math (Activation & Helpers)
# -------------------------------------------------------------
def relu(z):
    # The Bouncer: blocks negative vibes, passes positive signals
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # The Judge: turns energy scores into clean percentages (0% to 100%)
    z_max = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(y_true, probs):
    # The Penalty Score: how loud and wrong the guess was
    eps = 1e-15
    probs_clamped = np.clip(probs, eps, 1.0 - eps)
    return -np.mean(np.sum(y_true * np.log(probs_clamped), axis=1))

def init_network(hidden_width):
    # Hire apprentices: 64 inputs -> Layer 1 -> Layer 2 -> 10 mailboxes
    np.random.seed(42)
    W1 = np.random.normal(0, 0.01, (64, hidden_width))
    b1 = np.zeros(hidden_width)
    W2 = np.random.normal(0, 0.01, (hidden_width, hidden_width))
    b2 = np.zeros(hidden_width)
    W3 = np.random.normal(0, 0.01, (hidden_width, 10))
    b3 = np.zeros(10)
    return [W1, b1, W2, b2, W3, b3]

def forward(X_batch, params):
    # Forward Pass: the clue travels through the workshop to the judge
    W1, b1, W2, b2, W3, b3 = params
    z1 = X_batch @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    a3 = softmax(z3)
    cache = (X_batch, z1, a1, z2, a2, z3, a3)
    return a3, cache

def backward(y_batch, params, cache):
    # Backpropagation: walking backward through the room making corrections
    W1, b1, W2, b2, W3, b3 = params
    X_batch, z1, a1, z2, a2, z3, a3 = cache
    N = X_batch.shape[0]

    dz3 = (a3 - y_batch) / N
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0)

    da2 = dz3 @ W3.T
    dz2 = da2 * relu_derivative(z2)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)
    dW1 = X_batch.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return [dW1, db1, dW2, db2, dW3, db3]

# -------------------------------------------------------------
# 3. Sanity Checks: Untrained Penalty & 100% Probability Sums
# -------------------------------------------------------------
test_params = init_network(16)
initial_probs, _ = forward(X_train[:32], test_params)

row_sums = np.sum(initial_probs, axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Softmax row sum check failed!"
initial_loss = compute_loss(y_train[:32], initial_probs)
print(f"Untrained Penalty: {initial_loss:.4f} (Expected near ln(10) = 2.3026)")

# -------------------------------------------------------------
# 4. Lie Detector Test (Numerical Gradient Check)
# -------------------------------------------------------------
print("\n--- Running Lie Detector (Gradient Check) ---")
grad_params = init_network(16)
X_mini = X_train[:32]
y_mini = y_train[:32]

a3_mini, cache_mini = forward(X_mini, grad_params)
analytic_grads = backward(y_mini, grad_params, cache_mini)

all_coords = []
for p_id in [0, 2, 4]:
    mat = grad_params[p_id]
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            all_coords.append((p_id, (r, c)))

# Sample 10 random dials to nudge
np.random.seed(1)
selected_coords = [all_coords[i] for i in np.random.choice(len(all_coords), 10, replace=False)]

passed_checks = 0
eps = 1e-5

for i, (p_id, (r, c)) in enumerate(selected_coords):
    orig = grad_params[p_id][r, c]
    
    # Nudge slightly up
    grad_params[p_id][r, c] = orig + eps
    probs_p, _ = forward(X_mini, grad_params)
    lp = compute_loss(y_mini, probs_p)
    
    # Nudge slightly down
    grad_params[p_id][r, c] = orig - eps
    probs_m, _ = forward(X_mini, grad_params)
    lm = compute_loss(y_mini, probs_m)
    
    # Reset back to original
    grad_params[p_id][r, c] = orig
    
    num_grad = (lp - lm) / (2 * eps)
    ana_grad = analytic_grads[p_id][r, c]
    
    diff = abs(num_grad - ana_grad)
    denom = max(abs(num_grad) + abs(ana_grad), 1e-12)
    rel_error = diff / denom
    
    passes = rel_error < 1e-6 or (abs(num_grad) < 1e-9 and abs(ana_grad) < 1e-9)
    if passes:
        passed_checks += 1
    print(f"Dial {i+1:02d}: Math Formula={ana_grad:.6e}, Physical Nudge={num_grad:.6e} -> {'PASS' if passes else 'FAIL'}")

print(f"Lie Detector Result: {passed_checks}/10 Passed (Required: >= 9/10)")

# -------------------------------------------------------------
# 5. Training 6 Different Workshop Configurations (50 Rounds)
# -------------------------------------------------------------
widths = [16, 64]
lrs = [0.001, 0.01, 0.1]
epochs = 50
batch_size = 32

history = {}
accuracies = {}

print("\n--- Training the Workshops ---")
for W in widths:
    for lr in lrs:
        config_name = f"W={W}, lr={lr}"
        params = init_network(W)
        loss_history = []
        num_train = X_train.shape[0]
        
        for ep in range(epochs):
            perm = np.random.permutation(num_train)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            
            for b in range(0, num_train, batch_size):
                xb = X_shuffled[b:b+batch_size]
                yb = y_shuffled[b:b+batch_size]
                probs, cache = forward(xb, params)
                grads = backward(yb, params, cache)
                
                # Apply the adjustments
                for p in range(len(params)):
                    params[p] -= lr * grads[p]
            
            full_probs, _ = forward(X_train, params)
            loss_history.append(compute_loss(y_train, full_probs))
            
        history[config_name] = loss_history
        
        # Test accuracy on the final exam
        test_probs, _ = forward(X_test, params)
        y_pred = np.argmax(test_probs, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        acc = np.mean(y_pred == y_true_labels)
        accuracies[config_name] = acc
        print(f"Setup [{config_name:12s}] -> Final Penalty: {loss_history[-1]:.4f} | Exam Score: {acc*100:.2f}%")

# Create results folder safely
results_dir = "results" if not os.path.exists("lab02") else "lab02/results"
os.makedirs(results_dir, exist_ok=True)
plot_path = os.path.join(results_dir, "loss_curves.png")

# Draw and save the progress graph
plt.figure(figsize=(9, 6))
for label, curve in history.items():
    plt.plot(curve, label=f"{label} (Final Score: {accuracies[label]*100:.1f}%)")
plt.xlabel("Practice Round (Epoch)")
plt.ylabel("Mistake Penalty (Loss)")
plt.title("Lab 02: Neural Network Learning Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"\nSaved graph to: {plot_path}")

# -------------------------------------------------------------
# 6. Stopwatch Duel: Forward Guess vs. Backward Correction
# -------------------------------------------------------------
print("\n--- Stopwatch Duel (100 Sprints) ---")
bench_params = init_network(64)
bench_X = X_train[:32]
bench_y = y_train[:32]

# Warm-up lap
probs, cache = forward(bench_X, bench_params)
_ = backward(bench_y, bench_params, cache)

forward_times = []
backward_times = []

for _ in range(100):
    t0 = time.perf_counter()
    probs, cache = forward(bench_X, bench_params)
    forward_times.append(time.perf_counter() - t0)
    
    t0 = time.perf_counter()
    _ = backward(bench_y, bench_params, cache)
    backward_times.append(time.perf_counter() - t0)

med_fwd = np.median(forward_times) * 1000
med_bwd = np.median(backward_times) * 1000

print(f"Forward Pass Speed:  {med_fwd:.4f} ms")
print(f"Backward Pass Speed: {med_bwd:.4f} ms")
print(f"Backward is {med_bwd / med_fwd:.2f}x slower than Forward")