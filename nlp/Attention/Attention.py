import torch
from torch import nn

from nlp.Seq2Seq.Seq2Seq import SequenceMask


def MaskedSoftmax(X, valid_lens):
    """进行mask的softmax操作"""""
    if valid_lens is None:  # valid_len为None，说明不需要mask
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape  # batch_size, num_queries, num_keys
        if valid_lens.dim() == 1:  # 如果valid_lens是一个向量
            # valid_lens的长度应该与batch_size对应，而每个batch_size中有shape[1]个query
            # 因此需要将valid_lens复制shape[1]次
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:  # 如果valid_lens是一个矩阵，则矩阵的行数与batch_size对应，列数与每个batch_size中的query个数对应shape[1]
            # 此时直接将valid_lens转换为一个向量
            valid_lens = valid_lens.reshape(-1)
        # 只留最后一个维度（当前query与所有key的点积）
        # 将无效位置的值设置为一个很大的负数，这样在经过softmax操作后，这些值就会接近0
        X = SequenceMask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class AdditiveAttention(nn.Module):
    """
    加性注意力
    a(k, q) = w^T*tanh(W_k*k + W_q*q)
    k: key, q: query
    k(num,key_size)   q(num,query_size)
    W_k(key_size,num_hiddens)   W_q(query_size,num_hiddens)
    w_v(num_hiddens,1)
    """""
    
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        """
        queries(batch_size, num_queries, query_size)
        keys(batch_size, num_keys, key_size)
        values(batch_size, num_keys, value_size)
        valid_lens(batch_size, num_queries)->reshape(-1)->(batch_size*num_queries,)
        valid_lens(batch_size,)->repeat(queries.shape[1])->(batch_size*num_queries,)
        """""
        
        # 每条query和key映射到num_hiddens维
        queries, keys = self.W_q(queries), self.W_k(keys)
        # queries(batch_size, num_queries, num_hiddens)
        # keys(batch_size, num_keys, num_hiddens)
        # W_k*k + W_q*q 使用广播机制
        # 每条query都要与所有的key进行计算
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # features(batch_size, num_queries, num_keys, num_hiddens)
        scores = self.w_v(features).squeeze(-1)
        # scores(batch_size, num_queries, num_keys)
        attention_weights = MaskedSoftmax(scores, valid_lens)
        # torch.bmm对两个三维矩阵进行批量矩阵乘操作
        # attention_weights(batch_size, num_queries, num_keys)
        # values(batch_size, num_keys, value_size)
        # result(batch_size, num_queries, value_size)
        # 即每个query对所有key的value进行加权求和
        return torch.bmm(attention_weights, values)


# queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# # queries(batch_size=2, num_queries=1, query_size=20)
# # keys(batch_size=2, num_keys=10, key_size=2)
# # values的小批量，两个值矩阵是相同的
# values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
# valid_lens = torch.tensor([2, 6])

# attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,dropout=0.1)
# attention.eval()
# print(attention(queries, keys, values, valid_lens))


class DotProductAttention(nn.Module):
    """
    缩放点积注意力
    a(k, q) = k*q/sqrt(d)
    weights = softmax(a(k, q))
    k: key, q: query
    k(num,key_size)   q(num,query_size)
    """""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        queries(batch_size, num_queries, query_size=d)
        keys(batch_size, num_keys, key_size=d)
        values(batch_size, num_keys, value_size)
        valid_lens(batch_size, num_queries)->reshape(-1)->(batch_size*num_queries,)
        valid_lens(batch_size,)->repeat(queries.shape[1])->(batch_size*num_queries,)
        """""

        d = queries.shape[-1]
        # queries与keys矩阵转置批量乘   再除以sqrt(d)
        # (batch_size, num_queries, d) * (batch_size, d, num_keys)
        #  -> (batch_size, num_queries, num_keys)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(d).float())
        # scores(batch_size, num_queries, num_keys)
        attention_weights = MaskedSoftmax(scores, valid_lens)
        # attention_weights(batch_size, num_queries, num_keys)
        # values(batch_size, num_keys, value_size)
        # result(batch_size, num_queries, value_size)
        return torch.bmm(attention_weights, values)
