import torch
import torch.nn as nn

device = torch.device('mps')


class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.alpha = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def batch_norm(self, X, eps=1e-5, momentum=0.9):
        if not torch.is_grad_enabled():
            X_hat = (X -self.moving_mean) / torch.sqrt(self.moving_var + eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                mean = X.mean(dim=0, keepdim=True)
                var = ((X - mean) ** 2).mean(dim=0, keepdim=True)
            else:
                mean = X.mean(dim=(0, 2, 3), keepdim=True)
                var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
            
            X_hat = (X - mean) / torch.sqrt(var + eps)
            self.moving_mean = momentum * self.moving_mean + (1.0 - momentum) * mean
            self.moving_var = momentum * self.moving_var + (1.0 - momentum) * var
            Y = self.alpha * X_hat + self.beta
            return Y, self.moving_mean, self.moving_var

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        
        Y, self.moving_mean, self.moving_var = self.batch_norm(X, self.moving_mean, self.moving_var)

        return Y


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


# net = NiN()
# X = torch.randn(size=(1, 1, 224, 224))
# for blk in net.get_net():
#     X = blk(X)
#     print(blk.__class__.__name__, 'output shape:\t', X.shape)


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # Path 1
        self.__p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2
        self.__p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.__p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3
        self.__p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.__p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4
        self.__p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.__p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = torch.relu(self.__p1_1(x))
        p2 = torch.relu(self.__p2_2(torch.relu(self.__p2_1(x))))
        p3 = torch.relu(self.__p3_2(torch.relu(self.__p3_1(x))))
        p4 = torch.relu(self.__p4_2(self.__p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)
    


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.__b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.__b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.__b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.__b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.__b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.__net = nn.Sequential(self.__b1,
                                    self.__b2, self.__b3, self.__b4,
                                    self.__b5, nn.Linear(1024, 10))
        
    def forward(self, x):
        return self.__net(x)
    
    def get_net(self):
        return self.__net
    

# net = GoogLeNet()
# X = torch.randn(size=(1, 1, 96, 96))
# for blk in net.get_net():
#     X = blk(X)
#     print(blk.__class__.__name__, 'output shape:\t', X.shape)

# net = nn.Sequential(
#     nn.Conv2d(1,6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
#     nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
#     nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
#     nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
#     nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
#     nn.Linear(84, 10)
# )

# net = net.to(device)
# X = torch.randn(size=(1, 1, 28, 28), device=device)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y = Y + X
        return torch.relu(Y)
    

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                           nn.BatchNorm2d(64), nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        self.net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))
        
    def resnet_block(self, in_channels, out_channels, num_residuals, first_block=False):
        if first_block:
            assert in_channels == out_channels
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return blk
    
    def forward(self, x):
        return self.net(x)
    

# net = ResNet()
# net = net.to(device)
# X = torch.randn(size=(1, 1, 224, 224), device=device)
# for layer in net.net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)
