import torch
from torch import nn

from nlp.Transformer.Transformer import EncoderBlock


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """
    :param tokens_a: list of tokens
    :param tokens_b: list of tokens
    :return:
    tokens: [[cls],tokens_a,[sep],tokens_b,[sep]]
    segments: [0,0,0,1,1]
    """""
    tokens = ['[cls]'] + tokens_a + ['[sep]']
    segments = [0] * len(tokens)
    if tokens_b:
        tokens += tokens_b + ['[sep]']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


class BertEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout, max_len=1000, key_size=768, query_size=768, value_size=768, **kwargs):
        super(BertEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                                                      ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=True))

        # 创建一个形状为(1, max_len, num_hiddens)的张量，其值是从标准正态分布中随机生成的
        # 1：批量大小维度，设置为1，因为位置嵌入对批量中的所有输入都是相同的
        # max_len：模型可以处理的最大序列长度
        # num_hiddens：隐藏层的大小，与嵌入大小相同
        # 对于[0,i,:]，得到第i个位置的嵌入向量
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens=None):
        """
        :param tokens: (batch_size, seq_len)
        :param segments: (batch_size, seq_len)
        :param valid_lens: (batch_size,)
        :return X: (batch_size, seq_len, num_hiddens)
        """""
        # X (batch_size, seq_len, num_hiddens)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        # 取出前seq_len个位置的嵌入向量
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X


class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_hiddens, num_inputs), nn.ReLU(), nn.LayerNorm(num_inputs),
                                 nn.Linear(num_inputs, vocab_size))

    def forward(self, X, pred_positions):
        """
        X为encoder的输出(batch_size, seq_len, num_hiddens)在pred_positions位置上进行了mask
        :param X: (batch_size, seq_len, num_hiddens)
        :param pred_positions: (batch_size, num_pred_positions)  每个批次的预测位置
        返回每个预测位置的softmax输出
        :return mlm_Y_hat: (batch_size * num_pred_positions, vocab_size)
        """""
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]  # 取出mask位置的计算结果
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__()
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        """
        X为encoder输出的每个句子的第一个token([cls])的隐藏状态
        :param X: (batch_size, num_hiddens)
        :return: nsp_Y_hat: (batch_size, 2)
        """""
        return self.output(X)


# vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
# norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
# encoder = BertEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
#                       dropout)
#
# tokens = torch.randint(0, vocab_size, (2, 8))  # batch_size=2, seq_len=8  其中包含的整数值在 [0, vocab_size) 范围内
# segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])  # batch_size=2, seq_len=8
# encoded_X = encoder(tokens, segments)
#
# mlm = MaskLM(vocab_size, num_hiddens)
# mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])  # batch_size=2, num_pred_positions=3
# mlm_Y_hat = mlm(encoded_X, mlm_positions)
#
# mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])  # batch_size=2, num_pred_positions=3
# loss = nn.CrossEntropyLoss(reduction='none')  # reduction='none'表示不对损失函数求平均
# mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1))
#
# # (batch_size, seq_len, num_hiddens) -> (batch_size, seq_len*num_hiddens)  仅供测试，与实际用途不符
# encoded_X = torch.flatten(encoded_X, start_dim=1)
# nsp = NextSentencePred(encoded_X.shape[-1])
# nsp_Y_hat = nsp(encoded_X)
# nsp_Y = torch.tensor([0, 1])
# nsp_l = loss(nsp_Y_hat, nsp_Y)
#
# print(mlm_l, nsp_l)


class BertModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                 dropout, max_len=1000, key_size=768, query_size=768, value_size=768, hid_in_features=768,
                 mlm_in_features=768, nsp_in_features=768, **kwargs):
        super(BertModel, self).__init__()
        self.encoder = BertEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                                   num_layers, dropout, max_len, key_size, query_size, value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        """
        :param tokens: (batch_size, seq_len)
        :param segments: (batch_size, seq_len)
        :param valid_lens: (batch_size,)
        :param pred_positions: (batch_size, num_pred_positions)
        :return: encoded_X: (batch_size, seq_len, num_hiddens)  # encoder的输出
         mlm_Y_hat: (batch_size, num_pred_positions, vocab_size)  # 每个序列的预测位置的softmax输出
         nsp_Y_hat: (batch_size, 2)  # 每个序列中的两个句子是否连续的softmax输出
        """""
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None

        # 取出每个句子的第一个token([cls])，过两个线性层
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))

        return encoded_X, mlm_Y_hat, nsp_Y_hat


# vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
# norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
# Bert = BertModel(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
# tokens = torch.randint(0, vocab_size, (2, 8))  # batch_size=2, seq_len=8  其中包含的整数值在 [0, vocab_size) 范围内
# segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])  # batch_size=2, seq_len=8
# mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])  # batch_size=2, num_pred_positions=3
# res = Bert(tokens, segments, pred_positions=mlm_positions)
# for i in res:
#     if i is not None:
#         print(i.shape)
