import torch
from torch import nn


def SequenceMask(X, X_len, value=0):
    """X_len表示每行的有效长度 根据X_len处理X的每行 有效长度外设为value"""""
    maxlen = X.size(1)  # 是X的第二个维度大小，即时间步数
    # X_len是一个二维张量，第一维是批量大小，第二维是每个样本的长度

    # torch.arange(maxlen, dtype=torch.float32, device=X.device)`: 创建一个从`0`到`maxlen - 1`的一维张量，
    # 数据类型为`float32`，并且与`X`在同一个设备上。
    # `[None,:]`: 将一维张量扩展为二维张量，形状变为`(1, maxlen)` `X_len[:, None]`: 将`X_len` 张量扩展为二维张量，形状变为`(batch_size, 1)`。
    # ` < `: 比较两个张量，生成一个布尔掩码，形状为`(batch_size, maxlen)`。  广播机制
    # 掩码中每个位置的值表示该位置是否小于对应的`X_len`值。
    mask = torch.arange(maxlen, dtype=torch.float32, device=X.device)[None, :] < X_len[:, None]
    # [[0,1,2],     [[1,1,1],
    #  [0,1,2]]     [2,2,2]]
    # mask的形状是(批量大小, 时间步数)，其中第i行的前X_len[i]个元素为1，其余为0
    # `~mask`: 对掩码取反，得到一个新的布尔掩码，表示哪些位置的值需要被替换。
    X[~mask] = value
    return X


# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(SequenceMask(X, torch.tensor([1, 2])))
# mask = [[1,0,0], [1,1,0]]
# [[1,0,0], [4,5,0]]


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""""

    def forward(self, pred, label, valid_len):
        # `pred`的形状是(批量大小, 最大长度, 词表大小),最后一维是每个词的概率
        # `label`的形状是(批量大小, 最大长度)
        # `valid_len`的形状是(批量大小,)，每个元素是样本的有效长度
        weights = torch.ones_like(label)  # 先创建一个全1的权重张量
        weights = SequenceMask(weights, valid_len)  # 然后根据`valid_len`生成掩码
        self.reduction = 'none'  # 因为下面需要对每个位置的损失加权，所以这里指定损失函数的缩减方式为'none'
        # 调用PyTorch接口计算交叉熵损失，得到(batch_size, max_len)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        # 在句子级别上计算损失  mean之前是每个词的损失  得到(batch_size,)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss