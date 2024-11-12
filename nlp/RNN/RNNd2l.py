import math
import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F

from gb import get_device
from VocabandDataset.LMandDataset import load_data_time_machine

device = get_device()


def get_params(vocab_size, num_hiddens, device=device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附上梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    # `inputs`的形状：(`时间步数量`，`批量大小`，`词表大小`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # `X`的形状：(`批量大小`，`词表大小`)
    for X in inputs:  # 沿时间步迭代
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q  # `Y`的形状：(`批量大小`，`词表大小`)
        outputs.append(Y)  # outputs的形状：(`时间步数量`，`批量大小`，`词表大小`)
    return torch.cat(outputs, dim=0), (H,)  # 返回形状：(`时间步数量`*`批量大小`，`词表大小`), (H,)


class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


# X = torch.arange(10).reshape((2, 5))
# num_hiddens = 256
# net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
# state = net.begin_state(X.shape[0], device)
# Y, new_state = net(X.to(device), state)
# print(Y.shape, len(new_state), new_state[0].shape)


def predict_ch8(prefix, num_preds, net, vocab, device):
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


# net = RNNModelScratch(len(vocab), 512, device, get_params, init_rnn_state, rnn)
# print(predict_ch8('time traveller ', 10, net, vocab, device))


def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    # 初始化状态和计时器
    state = None
    # 初始化指标累加器，用于跟踪训练损失总和和标记数量
    # metric = d2l.Accumulator(2)  # [训练损失总和, 标记数量]
    metric = [0,0]
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
        if isinstance(updater, torch.optim.Optimizer):
            # 如果使用 PyTorch 优化器，清零梯度，执行反向传播，裁剪梯度，并更新参数
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            # 如果使用自定义更新器，执行反向传播，裁剪梯度，并更新参数
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)

        # 累加损失和标记数量
        # metric.add(l * y.numel(), y.numel())
        metric[0] += l * y.numel()
        metric[1] +=y.numel()
    # 返回困惑度和每秒标记数量
    return math.exp(metric[0] / metric[1])


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    # 初始化损失函数
    loss = nn.CrossEntropyLoss()
    # 初始化梯度下降算法
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练模型
    for epoch in range(num_epochs):
        # 使用困惑度作为评估指标
        ppl = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            # print(f'困惑度 {ppl:.1f}, {speed:.1f} 标记/秒 {str(device)}')
    print(f'最终困惑度 {ppl:.1f}, {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


def train1():
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_epochs, lr = 500, 0.1
    net = RNNModelScratch(len(vocab), 512, device, get_params, init_rnn_state, rnn)
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
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


def train2():
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_epochs, lr = 500, 0.1
    rnn_layer = nn.RNN(len(vocab), 512)
    net = RNNModel(rnn_layer, len(vocab))
    net = net.to(device)
    # predict_ch8('time traveller', 10, net, vocab, device)
    train_ch8(net, train_iter, vocab, lr, num_epochs, device)


if __name__ == '__main__':
    train1()
    # train2()
