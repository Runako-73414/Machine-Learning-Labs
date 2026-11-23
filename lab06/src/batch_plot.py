import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "lab06/results/batch_size_study.csv"
)

plt.figure(figsize=(8, 5))

plt.plot(
    df["batch_size"],
    df["throughput"],
    marker="o"
)

plt.xlabel(
    "Batch Size"
)

plt.ylabel(
    "Samples per Second"
)

plt.title(
    "Batch Size Study"
)

plt.grid(True)

plt.savefig(
    "lab06/results/batch_size_study.png"
)

plt.close()

print(
    "Batch plot saved."
)