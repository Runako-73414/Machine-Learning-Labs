import yaml
import torch
import torch.nn as nn

with open(
    "lab07/config.yaml",
    "r"
) as f:

    config = yaml.safe_load(f)

print(config)


class LargeCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                1,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(
                128 * 7 * 7,
                512
            ),

            nn.ReLU(),

            nn.Linear(
                512,
                10
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)


model = LargeCNN()

print(model)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    total_params
)

import torch

torch.save(
    model.state_dict(),
    "lab07/results/baseline_model.pth"
)

import os

size_bytes = os.path.getsize(
    "lab07/results/baseline_model.pth"
)

print(
    "Model size (bytes):",
    size_bytes
)

print(
    "Model size (MB):",
    size_bytes / (1024 * 1024)
)

import time

dummy_input = torch.randn(
    1,
    1,
    28,
    28
)

times = []

for _ in range(100):

    start = time.perf_counter()

    with torch.no_grad():

        model(dummy_input)

    end = time.perf_counter()

    times.append(
        end - start
    )

latency = (
    sum(times)
    / len(times)
)

print(
    "Latency:",
    latency
)