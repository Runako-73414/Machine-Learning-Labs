import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


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


levels = [
    0.10,
    0.25,
    0.50
]

for level in levels:

    model = LargeCNN()

    prune.ln_structured(
        model.network[0],
        name="weight",
        amount=level,
        n=2,
        dim=0
    )

    zeros = torch.sum(
        model.network[0].weight == 0
    )

    print(
        "Structured pruning:",
        level,
        "Zero weights:",
        zeros.item()
    )