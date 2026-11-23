import time
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def main():

    transform = transforms.ToTensor()

    dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    worker_values = [0, 1, 2, 4, 8]

    results = []

    for workers in worker_values:

        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
            num_workers=workers
        )

        start = time.perf_counter()

        for i, (images, labels) in enumerate(loader):

            if i >= 100:
                break

        end = time.perf_counter()

        elapsed = end - start

        results.append(
            [workers, elapsed]
        )

        print(
            "Workers:",
            workers,
            "Time:",
            elapsed
        )

    df = pd.DataFrame(
        results,
        columns=[
            "workers",
            "time"
        ]
    )

    df.to_csv(
        "lab06/results/worker_study.csv",
        index=False
    )

    print(
        "Worker study saved."
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["workers"],
        df["time"],
        marker="o"
    )

    plt.xlabel(
        "num_workers"
    )

    plt.ylabel(
        "Time (seconds)"
    )

    plt.title(
        "Worker Study"
    )

    plt.grid(True)

    plt.savefig(
        "lab06/results/worker_study.png"
    )

    plt.close()

    print(
        "Worker plot saved."
    )


if __name__ == "__main__":
    main()