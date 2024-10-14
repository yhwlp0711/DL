import torch
from torch import nn
import d2l.torch as d2l

from mnistSoftmax.softmaxRegression import evaluate_accuracy

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10
num_hiddens = 256
W1 = torch.normal(0, 0.01, size=(num_inputs, num_hiddens), requires_grad=True)
b1 = torch.zeros(num_hiddens, requires_grad=True)
W2 = torch.normal(0, 0.01, size=(num_hiddens, num_outputs), requires_grad=True)
b2 = torch.zeros(num_outputs, requires_grad=True)
params = [W1, b1, W2, b2]


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def sgd(batch_size=batch_size, lr=0.1):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def net(X):
    X = X.reshape((-1, num_inputs))
    H = torch.relu(X @ W1 + b1)
    return softmax(H @ W2 + b2)


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    mertics = d2l.Accumulator(3)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            l.backward()
            updater(X.shape[0])
            with torch.no_grad():
                mertics.add(l, d2l.accuracy(y_hat, y), y.numel())

        test_acc = evaluate_accuracy(net, test_iter)
        train_loss, train_acc = mertics[0] / mertics[2], mertics[1] / mertics[2]
        print(f'epoch {epoch + 1}, loss {train_loss:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')


num_epochs = 10
lr = 0.1


if __name__ == '__main__':
    train(net, train_iter, test_iter, cross_entropy, num_epochs, sgd)
    print('done')
