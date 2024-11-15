import collections
import math
import torch
from torch import nn

from nlp.RNN.RNN import grad_clipping
from nlp.Seq2Seq.EncoderDecoder import Encoder, Decoder, EncoderDecoder
from nlp.VocabandDataset.LoadTranslate import load_data_nmt, truncate_pad
from nlp.gb import get_device

device = get_device()


class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的编码器"""""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # X的形状是(批量大小, 时间步数)，转置后再获取词嵌入
        X = self.embedding(X)  # 输出形状是(批量大小, 时间步数, 词嵌入维度)
        X = X.permute(1, 0, 2)  # 输出形状是(时间步数, 批量大小, 词嵌入维度)
        output, state = self.rnn(X)  # output形状是(时间步数, 批量大小, 隐藏单元个数)  state形状是(层数, 批量大小, 隐藏单元个数)
        return output, state


class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的解码器"""""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # return enc_outputs[1]
        # 返回编码器的最后时刻每层的隐藏状态 以及 最后时刻最后一层的隐藏状态（用于做context）
        return enc_outputs[1], enc_outputs[1][-1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        # new
        encode = state[1]
        # state由tuple变为list
        state = state[0]
        # new end
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        # encode的形状:(batch_size,num_hiddens)
        # train过程只用了output，predict过程将（state，encode）赋值为dec_state并作为state传入
        # 则encode永远不变，故context也不变
        # 隐藏状态会改变
        return output, (state, encode)


# encoder = Seq2SeqEncoder(10, 8, 16, 2)
# decoder = Seq2SeqDecoder(10, 8, 16, 2)
# X = torch.zeros((4, 7), dtype=torch.long)
# state = decoder.init_state(encoder(X))
# output, state = decoder(X, state)
# print(output.shape, state.shape)


def SequenceMask(X, X_len, value=0):
    maxlen = X.size(1)  # 是X的第二个维度大小，即时间步数
    # X_len是一个二维张量，第一维是批量大小，第二维是每个样本的长度

    # torch.arange(maxlen, dtype=torch.float32, device=X.device)`: 创建一个从`0`到`maxlen - 1`的一维张量，
    # 数据类型为`float32`，并且与`X`在同一个设备上。
    # `[None,:]`: 将一维张量扩展为二维张量，形状变为`(1, maxlen)` `X_len[:, None]`: 将`X_len` 张量扩展为二维张量，形状变为`(batch_size, 1)`。
    # ` < `: 比较两个张量，生成一个布尔掩码，形状为`(batch_size, maxlen)`。  广播机制
    # 掩码中每个位置的值表示该位置是否小于对应的`X_len`值。
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < X_len[:, None]
    # mask的形状是(批量大小, 时间步数)，其中第i行的前X_len[i]个元素为1，其余为0
    # `~mask`: 对掩码取反，得到一个新的布尔掩码，表示哪些位置的值需要被替换。
    X[~mask] = value
    return X


# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(SequenceMask(X, torch.tensor([1, 2])))


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""""

    def forward(self, pred, label, valid_len):
        # `pred`的形状是(批量大小, 最大长度, 词表大小)
        # `label`的形状是(批量大小, 最大长度)
        # `valid_len`的形状是(批量大小,)，每个元素是样本的有效长度
        weights = torch.ones_like(label)  # 先创建一个全1的权重张量
        weights = SequenceMask(weights, valid_len)  # 然后根据`valid_len`生成掩码
        self.reduction = 'none'  # 因为下面需要对每个位置的损失加权，所以这里指定损失函数的缩减方式为'none'
        # 调用PyTorch接口计算交叉熵损失，得到一个三维张量
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        # 在句子级别上计算损失
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


# loss = MaskedSoftmaxCELoss()
# l = loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0]))
# print(l)


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""""

    def xavier_init_weights(m):
        """Xavier随机初始化"""""
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        # timer = d2l.Timer()
        # metric = d2l.Accumulator(2)
        metric = [0, 0]  # 训练损失总和，标记总数
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_vlen, Y, Y_vlen = [x.to(device) for x in batch]
            # 为了让输出对齐，如果不加bos，则输出会从第二个词开始，并最后多一个词
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 教师强制
            # return 了 output, (state, encode)->state即当前的state， encode即
            Y_hat, _ = net(X, dec_input, X_vlen)
            l = loss(Y_hat, Y, Y_vlen)
            l.sum().backward()  # 损失函数的标量进行反向传播
            grad_clipping(net, 1)
            num_tokens = Y_vlen.sum()
            optimizer.step()
            with torch.no_grad():
                metric[0] += l.sum()
                metric[1] += num_tokens
                # metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, loss {metric[0] / metric[1]:.3f}')
        #     animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}', f'{str(device)}')


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""""
    # 在预测时将net设置为评估模式
    net.eval()
    # 把源语言句子转为小写，并分词，再加上eos符号，然后转为词索引
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)  # 源语言句子有效长度
    # 填充或截断源语言句子，使其长度变为num_steps
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    # [num_steps] -> [1, num_steps]
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # encoder的输出Y和state，分别为每个时刻最后一层的状态以及最后时刻每层的状态，或许可以理解为源语言句子的内部表示
    # 随后decoder使用这个内部表示来生成目标语言句子
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # 虽然传入了Y和state，但是只取state作为decoder的初始状态
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    # [1] -> [1, 1]，为bos在词表中的索引，作为decoder的初始输入
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        #  根据encoder得到的内部表示和decoder的初始输入（bos），得到输出和新的状态
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重（稍后讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        # eos不会加入输出序列
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    """计算 BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i:i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i:i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i:i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def test():
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 300
    # train_iter返回四个张量：源语言句子、源语言句子有效长度、目标语言句子、目标语言句子有效长度
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device=device)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'"{eng}" => "{translation}", bleu {bleu(translation, fra, k=2):.3f}')
        print(attention_weight_seq)


test()
