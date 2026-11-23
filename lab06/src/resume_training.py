import torch
import torch.nn as nn
import torch.optim as optim


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


checkpoint = torch.load(
    "lab06/results/checkpoint.pth"
)

print(
    "Checkpoint loaded."
)

print(
    "Saved epoch:",
    checkpoint["epoch"]
)

model = CNN()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)

start_epoch = checkpoint["epoch"]

print(
    "Resuming from epoch:",
    start_epoch
)