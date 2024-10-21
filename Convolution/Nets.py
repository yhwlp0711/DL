import torch
import torch.nn as nn


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
    

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.__net = nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, 5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2), 
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
    )

    def forward(self, x):
        return self.__net(x)
