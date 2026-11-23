import torch

model = torch.nn.Linear(
    10,
    2
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

step_scheduler = (
    torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.1
    )
)

cosine_scheduler = (
    torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10
    )
)

plateau_scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer
    )
)

print("StepLR created")

print("CosineAnnealingLR created")

print("ReduceLROnPlateau created")