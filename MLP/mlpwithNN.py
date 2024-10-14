import torch
from torch import nn
from d2l import torch as d2l

from mnistSoftmax.softmaxRegression import evaluate_accuracy

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10
num_hiddens = 256

net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_hiddens), nn.ReLU(), nn.Linear(num_hiddens, num_outputs))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    mertics = d2l.Accumulator(3)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            updater.zero_grad()
            l.backward()
            updater.step()
            with torch.no_grad():
                mertics.add(l * len(y), d2l.accuracy(net(X), y), y.numel())

        test_acc = evaluate_accuracy(net, test_iter)
        train_loss, train_acc = mertics[0] / mertics[2], mertics[1] / mertics[2]
        print(f'epoch {epoch + 1}, loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')


epochs = 10


if __name__ == '__main__':
    train(net, train_iter, test_iter, loss, epochs, trainer)
