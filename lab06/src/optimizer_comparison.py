import torch
import torch.nn as nn

model = nn.Linear(
    10,
    2
)

adam = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

sgd = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9
)

print("Adam created")

print("SGD + momentum created")