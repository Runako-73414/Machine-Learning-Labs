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

    def forward(self, x):

        return self.network(x)


model = LargeCNN()

sparsity_levels = [
    0.00,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95
]

for sparsity in sparsity_levels:

    test_model = LargeCNN()

    prune.l1_unstructured(
        test_model.network[0],
        name="weight",
        amount=sparsity
    )

    zeros = torch.sum(
        test_model.network[0].weight == 0
    )

    print(
        "Sparsity:",
        sparsity,
        "Zero weights:",
        zeros.item()
    )