import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchinfo import summary
from thop import profile

# Fix random seed so results are reproducible
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# -------------------------------------------------------------
# 1. Dataset Preparation (Fashion-MNIST)
# -------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

print("\nDownloading and preparing Fashion-MNIST dataset...")
full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_set   = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Split 60k train into 55k train and 5k validation
train_set, val_set = random_split(full_train, [55000, 5000], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=128, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=128, shuffle=False)

# -------------------------------------------------------------
# 2. Define the Three Contenders
# -------------------------------------------------------------
# Model 1: MLP (Brute Force Scribe)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

# Model 2: CNN (Local Detective with Magnifying Glass)
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# Model 3: Attention / Vision Transformer (Global Observer)
class PatchAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 16 patches of 7x7 (49 pixels each)
        self.patch_proj = nn.Linear(49, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        B = x.shape[0]
        # Slice into 16 patches of 7x7
        patches = x.unfold(2, 7, 7).unfold(3, 7, 7)
        patches = patches.contiguous().view(B, 16, 49)
        tokens = self.patch_proj(patches)
        feats = self.transformer(tokens)
        pooled = feats.mean(dim=1)
        return self.classifier(pooled)

models = {
    "MLP": MLP().to(device),
    "CNN": CNN().to(device),
    "Attention": PatchAttention().to(device)
}

# -------------------------------------------------------------
# 3. Parameters, FLOPs, and Latency Benchmarks
# -------------------------------------------------------------
dummy_batch1 = torch.randn(1, 1, 28, 28).to(device)
dummy_batch128 = torch.randn(128, 1, 28, 28).to(device)

model_stats = {}

print("\n" + "="*55)
print(" ARCHITECTURAL PROFILES & BENCHMARKS ")
print("="*55)

for name, m in models.items():
    m.eval()
    # Count parameters
    info = summary(m, input_size=(1, 1, 28, 28), verbose=False)
    params = info.total_params
    
    # Profile FLOPs with thop
    macs, _ = profile(m, inputs=(dummy_batch1,), verbose=False)
    measured_flops = int(macs * 2)  # Convert MACs (multiply-adds) to individual FLOPs
    
    # Measure Latency Batch 1 (100 runs)
    with torch.no_grad():
        for _ in range(10): _ = m(dummy_batch1)  # Warm-up
        times_b1 = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = m(dummy_batch1)
            times_b1.append(time.perf_counter() - t0)
        lat_b1 = float(torch.tensor(times_b1).median() * 1000)

        # Measure Latency Batch 128 (20 runs)
        for _ in range(5): _ = m(dummy_batch128)  # Warm-up
        times_b128 = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = m(dummy_batch128)
            times_b128.append(time.perf_counter() - t0)
        lat_b128 = float(torch.tensor(times_b128).median() * 1000)
        
    model_stats[name] = {
        "params": params,
        "measured_flops": measured_flops,
        "lat_b1": lat_b1,
        "lat_b128": lat_b128
    }
    print(f"\n[{name}]")
    print(f"  • Parameters:       {params:,}")
    print(f"  • Measured FLOPs:   {measured_flops:,}")
    print(f"  • Latency (B=1):    {lat_b1:.3f} ms")
    print(f"  • Latency (B=128):  {lat_b128:.3f} ms")

# -------------------------------------------------------------
# 4. Hand-Calculated FLOPs for 1 Representative Layer
# -------------------------------------------------------------
print("\n" + "="*55)
print(" HAND-CALCULATED FLOPs (Ch. 4 Formulas) ")
print("="*55)
# Linear layer in MLP: 784 -> 256
hand_mlp_layer = 2 * 784 * 256
print(f"1. MLP Layer (Linear 784->256):")
print(f"   Formula: 2 * in * out = 2 * 784 * 256 = {hand_mlp_layer:,} FLOPs")

# Conv2d layer in CNN: Conv2d(1, 16, K=3, pad=1), output H=28, W=28
hand_cnn_layer = 2 * (3 * 3) * 1 * 16 * 28 * 28
print(f"2. CNN Layer (Conv2d 1->16, 3x3):")
print(f"   Formula: 2 * K * K * Cin * Cout * Hout * Wout = 2 * 9 * 1 * 16 * 28 * 28 = {hand_cnn_layer:,} FLOPs")

# Attention Layer: QKV projection (seq=16, d_model=64)
hand_attn_qkv = 2 * 16 * 64 * (3 * 64)
hand_attn_scores = 2 * (16 ** 2) * 64
print(f"3. Attention Layer (16 patches, d_model=64):")
print(f"   QKV Projections: 2 * seq * d_model * (3 * d_model) = {hand_attn_qkv:,} FLOPs")
print(f"   Attention Scores: 2 * seq^2 * d_model = {hand_attn_scores:,} FLOPs")

# -------------------------------------------------------------
# 5. Training Loop (5 Epochs Each, Adam, lr=1e-3)
# -------------------------------------------------------------
print("\n" + "="*55)
print(" TRAINING 5 EPOCHS EACH ")
print("="*55)
criterion = nn.CrossEntropyLoss()

for name, m in models.items():
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
    m.train()
    print(f"\nTraining {name}...")
    for epoch in range(5):
        total_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = m(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/5 - Training Loss: {avg_loss:.4f}")
    
    # Evaluate on the official 10k test set
    m.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            preds = m(x_b).argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)
    test_acc = (correct / total) * 100
    model_stats[name]["test_acc"] = test_acc
    print(f"  ==> {name} Final Test Accuracy: {test_acc:.2f}%")

# -------------------------------------------------------------
# 6. Save Trade-Off Plots
# -------------------------------------------------------------
os.makedirs("lab03/results", exist_ok=True)

# Plot 1: Accuracy vs Parameters
plt.figure(figsize=(7, 5))
for name, data in model_stats.items():
    plt.scatter(data["params"], data["test_acc"], s=140, label=name)
    plt.text(data["params"], data["test_acc"] + 0.35, f"{name} ({data['test_acc']:.1f}%)")
plt.xlabel("Parameter Count (Number of Dials)")
plt.ylabel("Test Accuracy (%)")
plt.title("Lab 03: Accuracy vs. Parameter Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("lab03/results/accuracy_vs_params.png", dpi=150)
plt.close()

# Plot 2: Accuracy vs Latency (Batch 1)
plt.figure(figsize=(7, 5))
for name, data in model_stats.items():
    plt.scatter(data["lat_b1"], data["test_acc"], s=140, label=name)
    plt.text(data["lat_b1"], data["test_acc"] + 0.35, f"{name} ({data['test_acc']:.1f}%)")
plt.xlabel("Inference Latency Batch 1 (ms)")
plt.ylabel("Test Accuracy (%)")
plt.title("Lab 03: Accuracy vs. Inference Latency (Batch 1)")
plt.grid(True)
plt.tight_layout()
plt.savefig("lab03/results/accuracy_vs_latency.png", dpi=150)
plt.close()

print("\nSuccess! Both trade-off plots saved to lab03/results/.")