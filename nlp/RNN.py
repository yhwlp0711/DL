import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from LMandDataset import load_data_time_machine
from gb import get_device

device = get_device()


class RNNModel(nn.Module):
    def __init__(self, vocab_size, rnn_layer=None):
        super(RNNModel, self).__init__()
        if rnn_layer is None:
            rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=512)
        else:
            self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:  # 如果是单向的
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将`Y`的形状改为(`时间步数`*`批量大小`, `隐藏单元数`)
        # 它的输出形状是 (`时间步数`*`批量大小`, `词表大小`)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens), device=device))


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def predict(prefix, num_preds, net, vocab, device):
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 预热期
    """预热期，将输入的前缀输入到模型中，获得模型的状态"""
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # 预测
    """预测期，将模型的状态作为输入，预测num_preds个字符"""
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    # 初始化状态和计时器
    state = None
    # 初始化指标累加器，用于跟踪训练损失总和和标记数量
    # metric = d2l.Accumulator(2)  # [训练损失总和, 标记数量]
    metric = [0, 0]
    # 遍历训练数据
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 如果使用随机抽样，每次读取一个新的随机小批量，不存在时序信息，需要重新初始化隐藏状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 分离状态以防止反向传播通过整个训练历史
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()

        # 将目标张量 Y 展平，并将 X 和 y 移动到指定设备
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)

        # 前向传播：计算预测输出和新的状态
        y_hat, state = net(X, state)

        # 计算损失
        # y.long()将y转换为整数类型，以便通过one_hot编码
        l = loss(y_hat, y.long()).mean()
        # 反向传播和参数更新
        # 如果使用 PyTorch 优化器，清零梯度，执行反向传播，裁剪梯度，并更新参数
        updater.zero_grad()
        l.backward()
        grad_clipping(net, 1)
        updater.step()

        # 累加损失和标记数量
        # metric.add(l * y.numel(), y.numel())
        metric[0] += l * y.numel()
        metric[1] += y.numel()

    # 返回困惑度和每秒标记数量
    return math.exp(metric[0] / metric[1])


def train(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    # 初始化损失函数
    loss = nn.CrossEntropyLoss()
    # 初始化梯度下降算法
    updater = torch.optim.SGD(net.parameters(), lr)
    predictd = lambda prefix: predict(prefix, 50, net, vocab, device)
    # 训练模型
    for epoch in range(num_epochs):
        # 使用困惑度作为评估指标
        ppl = train_epoch(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predictd('time traveller'))
            # print(f'困惑度 {ppl:.1f}, {speed:.1f} 标记/秒 {str(device)}')
    print(f'最终困惑度 {ppl:.1f}, {str(device)}')
    print(predictd('time traveller'))
    print(predictd('traveller'))
    

if __name__ == '__main__':
    train_iter, vocab = load_data_time_machine(batch_size=32, num_steps=35)
    vocab_size = len(vocab)
    num_hiddens = 512
    num_epochs, lr = 500, 1
    net = RNNModel(vocab_size, nn.RNN(vocab_size, num_hiddens))
    net = net.to(device)
    train(net, train_iter, vocab, lr, num_epochs, device)
    print(predict('time traveller', 50, net, vocab, device))
    print(predict('traveller', 50, net, vocab, device))
