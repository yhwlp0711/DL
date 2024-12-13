import collections
import re
import platform
# from d2l import torch as d2l

# d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')


def read_time_machine():
    """Load the time machine dataset into a list of text lines."""
    path = ''
    if platform.system() == 'Windows':
        path = 'C:/MO/CODE/Python/NLP/nlp/data/timemachine.txt'
    elif platform.system() == 'Linux':
        path = '/mnt/disk_8Td/zhn/father/NLP/nlp/data/timemachine.txt'
    elif platform.system() == 'Darwin':
        path = '/Users/chenmo/code/NLP/nlp/data/timemachine.txt'
    with open(path) as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# lines = read_time_machine()
# print(f'# text lines: {len(lines)}')
# print(lines[0])
# print(lines[10])


def tokenize(lines, token='word'):
    """
    将文本行拆分为单词或字符标记
    """""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)


# tokens = tokenize(lines)
# for i in range(11):
#     print(tokens[i])


def count_corpus(tokens):
    """返回一个字典，键是token，值是token的频率"""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        # tokens为二维列表，将其展平为一维列表
        tokens = [token for line in tokens for token in line]
    # 返回一个字典，键是token，值是token的频率
    return collections.Counter(tokens)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 统计词频
        counter = count_corpus(tokens)
        # 按照词频排序
        # counter.items() 返回一个包含字典所有键值对的列表
        # 排序时使用每个键值对的第二个值（即词频）作为排序的依据
        # reverse=True 表示降序排序
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # The index for the unknown token is 0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        # 仅保留词频大于min_freq的token
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            # 将token添加到idx_to_token列表中，根据token的索引可以找到token
            self.idx_to_token.append(token)
            # 将token添加到token_to_idx字典中，键是token，值是token的索引
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


# vocab = Vocab(tokens)


def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)  # Vocabn中的token按照频率排序
    corpus = [vocab[token] for line in tokens for token in line]  # 找到每个token在vocab中的索引
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# corpus, vocab = load_corpus_time_machine()
# a = 1
