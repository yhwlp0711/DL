import os
import re

import torch
from torch.utils import data

from nlp.VocabandDataset.LoadTranslate import truncate_pad
from nlp.VocabandDataset.Vocab import tokenize, Vocab


def read_snli(data_dir, is_train):
    """
    将SNLI数据集解析为前提、假设和标签
    :param data_dir: 数据集所在目录
    :param is_train: 是否为训练集
    
    :return: tuple: 包含前提、假设和标签的元组
    """""

    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    # 标签集合，将标签映射为整数
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    # 根据是否为训练集选择相应的文件
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt' if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        # 读取文件并按行分割，每行是一个样本
        rows = [row.split('\t') for row in f.readlines()[1:]]
    # 提取前提、假设和标签
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


class SNLIDataset(torch.utils.data.Dataset):
    """用于加载SNLI数据集的自定义数据集"""

    def __init__(self, dataset, num_steps, vocab=None):
        """
        初始化SNLIDataset类
        :param dataset: (tuple) 包含前提、假设和标签的元组
        :param num_steps: 每个序列的最大长度
        :param vocab: vocab
        """""
        self.num_steps = num_steps
        # 对前提和假设进行分词
        all_premise_tokens = tokenize(dataset[0])
        all_hypothesis_tokens = tokenize(dataset[1])
        # 如果没有提供词汇表，则创建一个新的词汇表
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens + all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        # 对前提和假设进行填充
        self.premises = self._pad(all_premise_tokens)  # 有num_前提行，其每行是对应原句子中，每个单词的索引，以及填充
        self.hypotheses = self._pad(all_hypothesis_tokens)
        # 将标签转换为张量
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        """
        对输入序列进行填充
        :param lines: 输入序列列表

        :return: 填充后的序列张量
        """""
        return torch.tensor([truncate_pad(self.vocab[line], self.num_steps, self.vocab['<pad>'])
                             for line in lines])

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的一个样本
        :param idx: 样本索引
        :return: tuple 包含前提、假设和标签的元组
        """""
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)


def load_data_snli(batch_size, num_steps=50):
    """
    下载SNLI数据集并返回数据迭代器和词表
    :param num_steps: 每个序列的最大长度

    :return: tuple 包含训练集迭代器、测试集迭代器和词汇表的元组
    """""
    num_workers = 0
    data_dir = './data/snli/'
    # 读取训练集和测试集数据  tuple(前提list、假设list、标签list)
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    # 创建训练集和测试集的数据集对象
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    # 创建数据加载器
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab


# # 加载数据并打印词汇表大小
# train_iter, test_iter, vocab = load_data_snli(128, 50)
# print(len(vocab))
#
# # 打印一个批次的数据形状
# for X, Y in train_iter:
#     print(X[0].shape)
#     print(X[1].shape)
#     print(Y.shape)
#     break
