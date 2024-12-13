import json
import multiprocessing
import os
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

from nlp.BERT.SNLIDataLoad import read_snli
from nlp.VocabandDataset.Vocab import Vocab, tokenize
from nlp.gb import get_device
from nlp.BERT.Bert import BertModel, get_tokens_and_segments

device = get_device()
# d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
#                              '225d66f04cae318b841a13d32af3acc165f253ac')
# d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
#                               'c72329e68a732bef0452e4b96a1c341c8910f81f')


def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, device):
    # data_dir = d2l.download_extract(pretrained_model)
    data_dir = os.path.join('./models/', pretrained_model)
    # 定义空词表以加载预定义词表
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    bert = BertModel(len(vocab), num_hiddens, norm_shape=[256],
                     ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                     num_heads=4, num_layers=2, dropout=0.2,
                     max_len=max_len, key_size=256, query_size=256,
                     value_size=256, hid_in_features=256,
                     mlm_in_features=256, nsp_in_features=256)
    # 加载预训练BERT参数
    bert.load_state_dict(torch.load(os.path.join(data_dir, 'pretrained.params')))
    return bert, vocab


class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[tokenize([s.lower() for s in sentences]) for sentences in dataset[:2]])]

        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 使用4个进程
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 为BERT输入中的'<CLS>'、'<SEP>'和'<SEP>'词元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)


class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))


def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices):
    """Train a model with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    # num_batches = len(train_iter)
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
    #                         legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        print('epoch:', epoch)
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        # metric = d2l.Accumulator(4)
        metric = [0.0, 0.0, 0, 0]
        for i, (features, labels) in enumerate(train_iter):
            # timer.start()
            # print('i:', i)
            l, acc = train_batch_ch13(net, features, labels, loss, trainer, devices)
            # metric.add(l, acc, labels.shape[0], labels.numel())
            metric[0] += l
            metric[1] += acc
            metric[2] += labels.shape[0]
            metric[3] += labels.numel()
            # timer.stop()
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (metric[0] / metric[2], metric[1] / metric[3],
            #                   None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        # animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
              f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    # print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
    #       f'{str(devices)}')


if __name__ == '__main__':
    # 如果出现显存不足错误，请减少“batch_size”。在原始的BERT模型中，max_len=512
    bert, vocab = load_pretrained_model('bert.small.torch', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
                                        num_layers=2, dropout=0.1, max_len=512, device=device)
    # print(bert)
    batch_size, max_len, num_workers = 512, 128, 0
    # data_dir = d2l.download_extract('SNLI')
    data_dir = './data/snli/'
    train_set = SNLIBERTDataset(read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDataset(read_snli(data_dir, False), max_len, vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size, num_workers=num_workers)
    net = BERTClassifier(bert)
    lr, num_epochs = 1e-4, 10
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices=[device])
