import time
import statistics
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(
    n_samples=500,
    noise=0.2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

print("Train shape:", X_train.shape)

print("Test shape:", X_test.shape)

class TinyNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(2, 8),

            nn.ReLU(),

            nn.Linear(8, 1),

            nn.Sigmoid()
        )

    def forward(self, x):

        return self.network(x)

model = TinyNet()

print(model)

X_train_t = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_t = torch.tensor(
    y_train.reshape(-1, 1),
    dtype=torch.float32
)

X_test_t = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_t = torch.tensor(
    y_test.reshape(-1, 1),
    dtype=torch.float32
)

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(200):

    optimizer.zero_grad()

    predictions = model(X_train_t)

    loss = criterion(
        predictions,
        y_train_t
    )

    loss.backward()

    optimizer.step()

with torch.no_grad():

    test_pred = model(X_test_t)

    predicted = (
        test_pred > 0.5
    ).float()

    accuracy = (
        predicted.eq(y_test_t)
        .float()
        .mean()
        .item()
    )

print(
    "Test Accuracy:",
    accuracy
)

times = []

for _ in range(5):

    start = time.perf_counter()

    model = TinyNet()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    for epoch in range(200):

        optimizer.zero_grad()

        predictions = model(X_train_t)

        loss = criterion(
            predictions,
            y_train_t
        )

        loss.backward()

        optimizer.step()

    end = time.perf_counter()

    times.append(end - start)

median_time = statistics.median(times)

print(
    "Median Training Time:",
    median_time
)