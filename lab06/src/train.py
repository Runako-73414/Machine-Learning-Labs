import yaml
import mlflow
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
import torch.optim as optim
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =====================================
# LOAD CONFIG
# =====================================

with open(
    "lab06/config.yaml",
    "r"
) as f:

    config = yaml.safe_load(f)

print(config)

# =====================================
# REPRODUCIBILITY
# =====================================

torch.manual_seed(
    config["seed"]
)

print(
    "Seed:",
    config["seed"]
)

# =====================================
# MLFLOW SETUP
# =====================================

mlflow.set_experiment(
    config["mlflow_experiment"]
)

mlflow.start_run()

mlflow.log_param(
    "seed",
    config["seed"]
)

mlflow.log_param(
    "batch_size",
    config["batch_size"]
)

mlflow.log_param(
    "epochs",
    config["epochs"]
)

mlflow.log_param(
    "learning_rate",
    config["learning_rate"]
)

# =====================================
# DATASET
# =====================================

transform = transforms.ToTensor()

train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Training batches:",
    len(train_loader)
)

# =====================================
# MODEL
# =====================================

class CNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(
                32 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(x)

model = CNN()

print(model)

# =====================================
# LOSS + OPTIMIZER
# =====================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=config["learning_rate"]
)

print(
    "Optimizer:",
    config["optimizer"]
)

# =====================================
# CHECKPOINT SETUP
# =====================================

os.makedirs(
    "lab06/results",
    exist_ok=True
)

checkpoint_path = (
    "lab06/results/checkpoint.pth"
)

print(
    "Checkpoint path:",
    checkpoint_path
)

# =====================================
# TRAINING LOOP
# =====================================

for epoch in range(
    config["epochs"]
):
  with profile(
    activities=[
        ProfilerActivity.CPU
    ]
  ) as prof:
    running_loss = 0.0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch+1}",
        "Loss:",
        average_loss
    )

    mlflow.log_metric(
        "loss",
        average_loss,
        step=epoch + 1
    )

print(
    prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=10
    )
)


# =====================================
# SAVE CHECKPOINT
# =====================================

torch.save(
    {
        "epoch": config["epochs"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },
    checkpoint_path
)

print(
    "Checkpoint saved."
)

# =====================================
# SAVE CHECKPOINT AS ARTIFACT
# =====================================

mlflow.log_artifact(
    checkpoint_path
)

# =====================================
# END RUN
# =====================================

mlflow.end_run()

print(
    "MLflow run completed."
)