import time
import pandas as pd

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

batch_sizes = [
    32,
    64,
    128,
    256,
    512
]

results = []

for batch_size in batch_sizes:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    start = time.perf_counter()

    samples = 0

    for i, (images, labels) in enumerate(loader):

        samples += len(images)

        if i >= 100:
            break

    end = time.perf_counter()

    throughput = (
        samples /
        (end - start)
    )

    print(
        "Batch:",
        batch_size,
        "Throughput:",
        throughput
    )

    results.append(
        [batch_size, throughput]
    )

df = pd.DataFrame(
    results,
    columns=[
        "batch_size",
        "throughput"
    ]
)

df.to_csv(
    "lab06/results/batch_size_study.csv",
    index=False
)

print(
    "Batch-size study saved."
)