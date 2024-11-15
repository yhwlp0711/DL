import os
import platform
import torch
from torch.utils import data
from d2l import torch as d2l

from nlp.VocabandDataset.Vocab import Vocab


# d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')


def read_data_nmt():
    """载入“英语－法语”数据集。"""""
    # data_dir = d2l.download_extract('fra-eng')
    path = ''
    if platform.system() == 'Windows':
        path = 'C:/MO/CODE/Python/NLP/nlp/data/fra-eng/fra.txt'
    elif platform.system() == 'Linux':
        path = '/mnt/disk_8Td/zhn/father/NLP/nlp/data/fra-eng/fra.txt'
    elif platform.system() == 'Darwin':
        path = ''
    with open(path, encoding="utf-8") as f:
        return f.read()


def preprocess_nmt(text):
    """预处理机器翻译数据集"""""
    def no_space(char, prev_char):
        """在单词和标点符号之间插入空格"""""
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # lower(): 将大写字母转换为小写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 如果前一个字符不是空格且当前字符是标点符号，则在当前字符前加空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化机器翻译数据集"""""
    source, target = [], []
    # 按行处理文本数据
    for i, line in enumerate(text.split('\n')):
        # 只处理 num_examples 个句子
        if num_examples and i > num_examples:
            break
        # 按制表符分割句子中的源语言和目标语言
        parts = line.split('\t')
        # 如果句子中没有制表符或者句子不包含两个翻译文本，则跳过
        if len(parts) == 2:
            # 按空格对源语言和目标语言进行词元化
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def get_vocab_nmt(source, target):
    """获取机器翻译数据集的词汇表"""""
    # 返回源语言和目标语言的词汇表
    return Vocab(source, min_freq=2, reserved_tokens=[
        '<pad>', '<bos>', '<eos>']), Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""""
    # 如果文本序列的长度大于 num_steps，则截断
    if len(line) > num_steps:
        return line[:num_steps]
    # 如果文本序列的长度小于 num_steps，则填充
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """将文本序列转换成小批量数据集"""""
    # 单词转为下标    lines: 二维列表
    lines = [vocab[l] for l in lines]
    # 每行添加结束符
    lines = [l + [vocab['<eos>']] for l in lines]
    # 每行进行截断或填充   array: 二维张量   num_steps: 每行的长度
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    # valid_len表示每行的有效长度    valid_len: 一维列表
    # != 返回一个与array相同形状的张量，值为0或1，0表示array中的元素等于vocab['<pad>']，1表示不等于
    # sum(1)表示按行求和，即每行的有效长度
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=1000):
    """返回翻译数据集的迭代器和词汇表"""""
    # 读取数据
    text = read_data_nmt()
    # 词元化：将文本数据转换为二维列表
    source, target = tokenize_nmt(preprocess_nmt(text), num_examples)
    # 获取词汇表
    src_vocab, tgt_vocab = get_vocab_nmt(source, target)
    # 构建数据集
    # 每句话都需要截断或填充到相同长度
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 转为DataLoader迭代器
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = data.DataLoader(data.TensorDataset(*data_arrays), batch_size, shuffle=True)
    # data_iter = d2l.load_array(data_arrays, batch_size)
    # 返回迭代器和词汇表  迭代器每次迭代返回四个张量：源语言句子、源语言句子有效长度、目标语言句子、目标语言句子有效长度
    return data_iter, src_vocab, tgt_vocab


def test():
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('valid lengths for X:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('valid lengths for Y:', Y_valid_len)
        break


# test()
