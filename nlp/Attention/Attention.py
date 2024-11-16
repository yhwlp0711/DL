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
