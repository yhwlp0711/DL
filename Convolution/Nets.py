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
            nn.Linear(16 * 5 * 5, 120),
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
            nn.Linear(256 * 5 * 5, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        return self.__net(x)


class VGG(nn.Module):
    def __init__(self, ):
        super(VGG, self).__init__()
        self.__conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        self.__net = self.vgg()

    def __vgg_block(self, num_convs, in_channels=1, out_channels=64):
        layers = []
        for num in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def vgg(self):
        conv_blks = []
        in_channels = 1
        for (num_convs, out_channels) in self.__conv_arch:
            conv_blks.append(self.__vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        return nn.Sequential(*conv_blks, nn.Flatten(), nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4096, 10))

    def forward(self, x):
        return self.__net(x)

    def get_net(self):
        return self.__net


# net = VGG()
# X = torch.randn(size=(1, 1, 224, 224))
# for blk in net.get_net():
#     X = blk(X)
#     print(blk.__class__.__name__, 'output shape:\t', X.shape)


class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        self.__net = nn.Sequential(
            self.nin_block(1, 96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            self.nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            self.nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            self.nin_block(384, 10, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.__net(x)

    def get_net(self):
        return self.__net


net = NiN()
X = torch.randn(size=(1, 1, 224, 224))
for blk in net.get_net():
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)