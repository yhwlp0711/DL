import random
import torch

from gb import get_device
from Vocab import read_time_machine, Vocab, load_corpus_time_machine
from d2l import torch as d2l


def get_vocab(data=read_time_machine()):
    tokens = d2l.tokenize(data)
    corpus = [token for line in tokens for token in line]
    vocab = Vocab(corpus)
    return vocab


def get_biggram_vocab(data=read_time_machine()):
    tokens = d2l.tokenize(data)
    corpus = [token for line in tokens for token in line]
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    return bigram_vocab


def get_trigram_vocab(data=read_time_machine()):
    tokens = d2l.tokenize(data)
    corpus = [token for line in tokens for token in line]
    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    return trigram_vocab


# a1 = get_vocab()
# a2 = get_biggram_vocab()
# a3 = get_trigram_vocab()
# print(a1.token_freqs[:10])
# print(a2.token_freqs[:10])
# print(a3.token_freqs[:10])


def seq_data_iter_random(corpus, batch_size, num_steps):
    # 从随机位置开始截取序列，确保序列长度至少为 num_steps
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 计算可以生成的子序列数量
    num_subseqs = (len(corpus) - 1) // num_steps
    # 生成初始索引列表，每个索引表示一个子序列的起始位置
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 打乱初始索引列表，确保数据随机性
    random.shuffle(initial_indices)

    # 定义一个内部函数，用于根据位置提取子序列
    def data(pos):
        return corpus[pos:pos + num_steps]

    # 计算可以生成的批次数量
    """有num_subseqs个子序列，每batch_size个子序列为一个批次"""
    num_batches = num_subseqs // batch_size
    # 迭代生成每个批次的数据
    for i in range(0, batch_size * num_batches, batch_size):
        # 获取当前批次的初始索引
        """每次取batch_size个子序列，即一个批次"""
        initial_indices_per_batch = initial_indices[i:i + batch_size]
        # 根据初始索引提取输入数据 X 和目标数据 Y
        """每次取出num_steps个元素，取batch_size次"""
        X = [data(j) for j in initial_indices_per_batch]
        """取出num_steps个元素，取batch_size次，每个元素后移一个位置"""
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # 将数据转换为 PyTorch 张量并返回
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    # 从随机位置开始截取序列，确保序列长度至少为 num_steps
    offset = random.randint(0, num_steps-1)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1:offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i:i + num_steps]
        Y = Ys[:, i:i + num_steps]
        yield X, Y


class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter=False, max_tokens=-1):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

    def __len__(self):
        num_batches = len(self.corpus) // self.batch_size
        return num_batches // self.num_steps * self.num_steps


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=-1):
    data_loader = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_loader, data_loader.vocab
