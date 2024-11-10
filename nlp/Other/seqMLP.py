import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

from gb import get_device

device = get_device()

T = 1000  # Generate a total of 1000 points

time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
# d2l.plt.show()

tau = 4
features = torch.zeros((T - tau, tau))  # 生成特征
for i in range(tau):
    features[:, i] = x[i: T - tau + i]  # 第i列的元素是x的第i到T-tau+i-1个元素
labels = x[tau:].reshape((-1, 1))  # 标签

batch_size, n_train = 16, 600
train_iter = data.DataLoader(data.TensorDataset(features[:n_train], labels[:n_train]), batch_size, shuffle=True)


def init_weights(m):
    if type(m) is nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net


def train(net, train_iter, epochs, lr=0.01):
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr)
    loss = nn.MSELoss()
    net.train()
    for epoch in range(epochs):
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        print(f'epoch {epoch + 1}, loss {float(l.mean()):.6f}')


net = get_net()
train(net, train_iter, 5)
y_pred = net(features.to(device)).detach().cpu()
d2l.plot([time, time[tau:]], [x, y_pred], 'time', 'x', legend=['data', 'pred'], xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()
