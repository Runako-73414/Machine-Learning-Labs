import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print("Training samples:", len(train_dataset))
print("Testing samples:", len(test_dataset))

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))

import torch.nn as nn

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

class DeepMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

class AttentionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.patch_embed = nn.Linear(49, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1
        )

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):

        batch_size = x.size(0)

        x = x.unfold(2, 7, 7).unfold(3, 7, 7)

        x = x.contiguous().view(batch_size, 16, 49)

        x = self.patch_embed(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.classifier(x)

        return x


model = AttentionModel()

print(model)

criterion = nn.CrossEntropyLoss()

print("Loss function created")

import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Optimizer created")

losses = []

for epoch in range(1, 6):

    running_loss = 0.0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print("Epoch", epoch, "Loss:", running_loss / len(train_loader))
    losses.append(running_loss / len(train_loader)) 

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print("Test Accuracy:", test_accuracy)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Total Parameters:", total_params)

from thop import profile

dummy = torch.randn(1, 1, 28, 28)

flops, params = profile(
    model,
    inputs=(dummy,),
    verbose=False
)

print("THOP FLOPs:", flops)
print("THOP Params:", params)

import time

sample = next(iter(test_loader))[0][:1]
sample128 = next(iter(test_loader))[0]

times = []

for _ in range(100):

    start = time.perf_counter()

    with torch.no_grad():
        model(sample)

    end = time.perf_counter()

    times.append(end - start)

latency_batch1 = sum(times) / len(times)

print("Batch 1 Latency:", latency_batch1)

sample128 = next(iter(test_loader))[0]

times128 = []

for _ in range(100):

    start = time.perf_counter()

    with torch.no_grad():
        model(sample128)

    end = time.perf_counter()

    times128.append(end - start)

latency_batch128 = sum(times128) / len(times128)

print("Batch 128 Latency:", latency_batch128)

plt.figure(figsize=(8, 5))

plt.plot(range(1, 6), losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Attention Model Loss Curve")

plt.grid(True)

plt.savefig("lab03/results/attention_loss_curve.png")

plt.close()

plt.figure(figsize=(8, 5))

models = ["MLP", "DeepMLP", "CNN", "Attention"]
accuracies = [86.38, 86.44, 90.56, 84.69]

plt.bar(models, accuracies)

plt.ylabel("Accuracy (%)")
plt.title("Architecture Accuracy Comparison")

plt.savefig("lab03/results/accuracy_comparison.png")

plt.close()