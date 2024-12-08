import torch
from d2l import torch as d2l
from torch import nn

from nlp.Attention.Attention import DotProductAttention
from nlp.Transformer.Positional import PositionalEncoding
from nlp.gb import get_device


def transpose_qkv(X, num_heads):
    """
    X (batch_size, num_queries or num_keys, num_hiddens)
    return X(batch_size*num_heads, num_queries or num_keys, num_hiddens/num_heads)
    """""
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # X (batch_size, num_queries or num_keys, num_heads, num_hiddens/num_heads)

    X = X.permute(0, 2, 1, 3)
    # X (batch_size, num_heads, num_queries or num_keys, num_hiddens/num_heads)

    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """
    X (batch_size*num_heads, num_queries or num_keys, num_hiddens/num_heads)
    return X(batch_size, num_queries or num_keys, num_hiddens)
    """""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 3, 1)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """
    
    """""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


num_hiddens, num_heads = 100, 5
# query_size = key_size = value_size = num_hiddens
# key_size  query_size  value_size  num_hiddens
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
attention.eval()
batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
# X: query, Y: key, Y: value
# return (batch_size,num_queries,value_size=key_size=num_hiddens)
res = attention(X, Y, Y, valid_lens)
print(res.shape)
