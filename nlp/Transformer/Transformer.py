import math

import torch
from torch import nn

from nlp.EncoderDecoder.EncoderDecoder import EncoderDecoder
from nlp.Seq2Seq.Seq2Seq import train_seq2seq, predict_seq2seq, bleu
from nlp.Transformer.MultiHeadAttention import MultiHeadAttention
from nlp.Transformer.Positional import PositionalEncoding
from nlp.VocabandDataset.LoadTranslate import load_data_nmt
from nlp.gb import get_device


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ln = nn.LayerNorm(2)
# bn = nn.BatchNorm1d(2)
# X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# # X (2, 2)  batchsize=2, num_features=2            feature1  feature2
# #                                         batch1: [1,             2]
# #                                         batch2: [2,             3]
# # ln: [[-1, 1], [-1, 1]]    针对每个样本的所有特征进行归一化
# # bn: [[-1, -1],            针对所有样本的每个特征分别进行归一化
# #       [1, 1]]


class PositionWiseFFN(nn.Module):
    """
    两层全连接网络
    """""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    """
    残差连接后的LayerNorm
    """""

    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """
    input:X
    Y = MultiHeadAttention(key:X, query:X, value:X)
    Y = AddNorm(X, Y)
    Y = PositionWiseFFN(Y)
    Y = AddNorm(Y, Y)
    output:Y
    """""
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    """
    input:X
    X = Embedding(X)
    X = PositionalEncoding(X)
    for EncoderBlock in EncoderBlocks:
        X = EncoderBlock(X)
    output:X
    """""
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.attention_weights = [None] * num_layers
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                                              ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))  # 位置编码
        # embedding的权重是一个服从均值为0，方差为1的正态分布，乘以math.sqrt(self.num_hiddens)后，其方差变为self.num_hiddens
        # PositionalEncoding内部需要做加法，所以需要保证embedding的输出和PositionalEncoding的输出的范围相差不大
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


# encoder = TransformerEncoder(200, 24, 24, 24, 24, [24],
#                              24, 48, 8, 2, 0.5)
# encoder.eval()
# X = torch.ones((2, 100), dtype=torch.long)  # batch_size=2, num_steps=100
# res = encoder(X, valid_lens=None)
# print(res.shape)  # torch.Size([2, 100, 24])


class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__()
        self.i = i  # layer number
        self.attention1 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # state: enc_outputs, enc_valid_lens, [None]*num_layers
        # X: (batch_size, num_steps, num_hiddens)
        # enc_outputs: (batch_size, num_steps, num_hiddens)
        # enc_valid_lens: (batch_size,)
        if state[2][self.i] is None:
            # 如果 state[2][self.i] 为 None，表示这是当前层第一次处理输入 X，因此 key_values 被设置为 X。
            key_values = X
        else:
            # 否则，key_values 将包含之前的状态和当前输入 X 的拼接结果。
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.attention_weights = [None] * num_layers
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input,
                                              ffn_num_hiddens, num_heads, dropout, i))

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self.attention_weights[i] = blk.attention1.attention.attention_weights
        return X, state

    @property
    def attention_weights_list(self):
        return self.attention_weights


def test():
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 100, get_device()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

    encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
                                 ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
                                 ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)

    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']

    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')


test()
