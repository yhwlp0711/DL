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


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.__net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(256*5*5, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        return self.__net(x) 