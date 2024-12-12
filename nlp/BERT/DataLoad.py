import os
import random
import torch
from torch.utils import data

from nlp.BERT.Bert import get_tokens_and_segments
from nlp.VocabandDataset.Vocab import Vocab, tokenize


def _read_wiki(data_dir):
    """
    读取整个数据集
    :return: paragraphs(num_paragraphs, num_sentences per paragraphs)  元素为string
    """""
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    # 每行代表一个段落，其中在任意标点符号及其前面的词元之间插入空格。保留至少有两句话的段落
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def _get_next_sentence(sentence, next_sentence, paragraphs):
    """
    生成二分类任务的训练样本
    50%返回原始sentence、next_sentence、True
    50%返回sentence、paragraphs中随机句子、False
    :param paragraphs: list of list of string
    """""
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        # paragraphs中随机抽一段，再随机抽一句
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """
    :param paragraph: 段落
    :param paragraphs: 全体段落
    :param vocab: Vocab
    :param max_len: tokens的最大长度
    :return: nsp_data_from_paragraph: list of tuple(拼接后的句子, segments, is_next)
    """""
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        # 对于每个段落，排除最后一句，其他的都做下一句预测任务
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        # 拼接两个句子的词元并得到segments
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    """
    :param tokens: list of 词元
    :param candidate_pred_positions: list of 非特殊词元的位置
    :param num_mlm_preds: 预测的词元数量
    :param vocab: Vocab
    :return: mlm_input_tokens: 遮蔽之后的list of 词元
             pred_positions_and_labels: list of tuple(遮蔽位置, 该位置的label)
    """""
    # 为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    mlm_input_tokens = [token for token in tokens]  # tokens的副本
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        # mlm_pred_position是当前词元索引
        if len(pred_positions_and_labels) >= num_mlm_preds:
            # 预测的词元数量达到num_mlm_preds时停止
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10%的时间：保持词不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10%的时间：用随机词替换该词
            else:
                # 词表中的随机词
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token  # 替换
        # 保存位置和标签
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    """
    :param tokens: list of 词元
    :param vocab: Vocab
    :return: vocab[mlm_input_tokens]: 遮蔽之后的list of 词元在词表中的索引
             pred_positions: 遮蔽的位置
             vocab[mlm_pred_labels]: 遮蔽的label在词表中的索引
    """""
    candidate_pred_positions = []  # 保存tokens中非特殊词元的索引
    # tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)

    # 遮蔽语言模型任务中预测15%的随机词元，round为取整
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)

    # 按照词元索引进行排序
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    """
    对BERT输入进行填充，使其具有相同的长度，并生成相应的遮蔽语言模型(MLM)和下一句预测(NSP)任务的标签和权重。
    :param examples: list of tuple(token_ids, pred_positions, mlm_pred_label_ids, segments, is_next)
                     token_ids: list of int, 词元在词表中的索引
                     pred_positions: list of int, 遮蔽词元的位置
                     mlm_pred_label_ids: list of int, 遮蔽词元的标签在词表中的索引
                     segments: list of int, 句子片段标识
                     is_next: int, NSP任务的标签，0表示下一句，1表示随机句
    :param max_len: int, tokens的最大长度
    :param vocab: Vocab, 词表对象
    :return: tuple of tensors:
            all_token_ids(num_tokens, max_len), all_segments(num_tokens, max_len),
            valid_lens(num_tokens), all_pred_positions(num_tokens, max_num_mlm_preds),
            all_mlm_weights(num_tokens, max_num_mlm_preds), all_mlm_labels(num_tokens, max_num_mlm_preds),
            nsp_labels(num_tokens)
    """""
    # 遮蔽语言模型(MLM)任务的最大预测词元数量
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        # 填充token_ids到max_len长度
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
                max_len - len(token_ids)), dtype=torch.long))
        # 填充segments到max_len长度
        all_segments.append(torch.tensor(segments + [0] * (
                max_len - len(segments)), dtype=torch.long))
        # valid_lens不包括'<pad>'的计数
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        # 填充pred_positions到max_num_mlm_preds长度，填充部分为0
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
                max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 填充mlm_weights到max_num_mlm_preds长度，填充部分权重为0，其实就相当于valid_lens
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                    max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        # 填充mlm_labels到max_num_mlm_preds长度
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
                max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        # 添加nsp任务的标签
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(torch.utils.data.Dataset):
    """
        all_token_ids(num_tokens, max_len), all_segments(num_tokens, max_len),
        valid_lens(num_tokens), all_pred_positions(num_tokens, max_num_mlm_preds),
        all_mlm_weights(num_tokens, max_num_mlm_preds), all_mlm_labels(num_tokens, max_num_mlm_preds),
        nsp_labels(num_tokens)
    """""
    def __init__(self, paragraphs, max_len):
        """
        初始化_WikiTextDataset类，处理输入段落并生成BERT模型所需的输入数据。

        :param paragraphs: list of list of str, 输入段落，每个段落是句子字符串列表
        :param max_len: int, tokens的最大长度
        """""
        # 输入paragraphs为二维列表，元素为句子；
        # 而输出paragraphs为三维列表，元素为单词
        paragraphs = [tokenize(paragraph, token='word') for paragraph in paragraphs]
        # 句子列表，即合并paragraphs前两个维度
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        # 根据句子创建vocab，仅保留词频大于或等于5的词元
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        # 获取下一句子预测任务的数据
        examples = []  # list of tuple(tokens, segments, is_next)
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))  # extend还是二维列表
        # 根据(tokens, segments, is_next) 获取遮蔽语言模型任务的数据
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next))
                    for tokens, segments, is_next in examples]
        # examples: list of tuple(token_ids, pred_positions, mlm_pred_label_ids, segments, is_next)
        # 填充输入
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的一个样本。

        :param idx: int, 样本索引
        :return: tuple of tensors,
                 包含对应索引的token_ids, segments, valid_lens, pred_positions, 
                 mlm_weights, mlm_labels, nsp_labels
        """""
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集"""
    # num_workers = d2l.get_dataloader_workers()
    # data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    data_dir = './wikitext-2/'
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True, num_workers=0)
    return train_iter, train_set.vocab


# batch_size, max_len = 512, 64
# train_iter, vocab = load_data_wiki(batch_size, max_len)
#
# for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
#      mlm_Y, nsp_y) in train_iter:
#     print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
#           pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
#           nsp_y.shape)
#     break
