import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(
    10,
    2
)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.MSELoss()

accumulation_steps = 4

print(
    "Batch size: 64"
)

print(
    "Accumulation steps:",
    accumulation_steps
)

print(
    "Effective batch size:",
    64 * accumulation_steps
)

for step in range(8):

    x = torch.randn(
        64,
        10
    )

    y = torch.randn(
        64,
        2
    )

    predictions = model(x)

    loss = criterion(
        predictions,
        y
    )

    loss = loss / accumulation_steps

    loss.backward()

    if (
        (step + 1)
        % accumulation_steps
        == 0
    ):

        optimizer.step()

        optimizer.zero_grad()

        print(
            "Optimizer step at iteration",
            step + 1
        )

print(
    "Gradient accumulation completed."
)