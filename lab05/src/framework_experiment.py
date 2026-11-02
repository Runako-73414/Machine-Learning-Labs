import torch
import torch.nn as nn
import time
import statistics

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

x = torch.randn(10, 2)

output = model(x)

print("Shape:", x.shape)

print("Dtype:", x.dtype)

print("Contiguous:", x.is_contiguous())

eager_times = []

for _ in range(5):

    start = time.perf_counter()

    for _ in range(1000):
        model(x)

    end = time.perf_counter()

    eager_times.append(end - start)

eager_median = statistics.median(
    eager_times
)

print(
    "Eager median:",
    eager_median
)

traced_model = torch.jit.trace(
    model,
    x
)

traced_times = []

for _ in range(5):

    start = time.perf_counter()

    for _ in range(1000):
        traced_model(x)

    end = time.perf_counter()

    traced_times.append(end - start)

traced_median = statistics.median(
    traced_times
)

print(
    "Traced median:",
    traced_median
)

large = torch.randn(1000, 1000)

non_contiguous = large.t()

print(
    "\nNon-contiguous:",
    non_contiguous.is_contiguous()
)

contiguous_times = []

for _ in range(5):

    start = time.perf_counter()

    for _ in range(100):
        torch.sum(large)

    end = time.perf_counter()

    contiguous_times.append(
        end - start
    )

contiguous_median = statistics.median(
    contiguous_times
)

print(
    "Contiguous median:",
    contiguous_median
)

non_contiguous_times = []

for _ in range(5):

    start = time.perf_counter()

    for _ in range(100):
        torch.sum(non_contiguous)

    end = time.perf_counter()

    non_contiguous_times.append(
        end - start
    )

non_contiguous_median = statistics.median(
    non_contiguous_times
)

print(
    "Non-contiguous median:",
    non_contiguous_median
)