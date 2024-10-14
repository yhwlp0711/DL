import torch
from torch import nn
import d2l.torch as d2l
from softmaxRegression import evaluate_accuracy

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10

net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs), nn.Softmax(dim=1))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
loss = nn.CrossEntropyLoss()  # softmax first, then cross entropy
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 50


def train_epoch(net, train_iter, loss, updater):
    net.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        temp = y_hat.sum(axis=1)
        updater.zero_grad()
        l.backward()
        updater.step()
        metric.add(float(l) * len(y), d2l.accuracy(y_hat, y), y.numel())

    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        print(f'epoch {epoch + 1}, train loss {train_metrics[0]:.6f}, train acc {train_metrics[1]:.6f}')
    test_acc = evaluate_accuracy(net, test_iter)
    print(f'test acc {test_acc:.6f}')
    train_loss, train_acc = train_metrics
    return train_loss, train_acc, test_acc


if __name__ == '__main__':
    train(net, train_iter, test_iter, loss, num_epochs, trainer)
