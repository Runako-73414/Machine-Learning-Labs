import torch
import torch.nn as nn


class TeacherCNN(nn.Module):

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


teacher = TeacherCNN()

teacher_params = sum(
    p.numel()
    for p in teacher.parameters()
)

print(
    "Teacher parameters:",
    teacher_params
)


class StudentCNN(nn.Module):

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
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                10
            )
        )

    def forward(self, x):

        return self.network(x)


student = StudentCNN()

student_params = sum(
    p.numel()
    for p in student.parameters()
)

print(
    "Student parameters:",
    student_params
)

temperature = 4

alpha = 0.5

print(
    "Temperature:",
    temperature
)

print(
    "Alpha:",
    alpha
)

teacher_logits = torch.randn(
    8,
    10
)

student_logits = torch.randn(
    8,
    10
)

labels = torch.randint(
    0,
    10,
    (8,)
)

ce_loss = nn.CrossEntropyLoss()(
    student_logits,
    labels
)

kl_loss = nn.KLDivLoss(
    reduction="batchmean"
)(
    torch.log_softmax(
        student_logits / temperature,
        dim=1
    ),

    torch.softmax(
        teacher_logits / temperature,
        dim=1
    )
)

distillation_loss = (

    alpha * ce_loss

    +

    (1 - alpha)

    * (temperature ** 2)

    * kl_loss
)

print(
    "Distillation Loss:",
    distillation_loss.item()
)