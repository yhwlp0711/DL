import torch
from torch import nn

from nlp.EncoderDecoder.EncoderDecoder import Encoder, Decoder, EncoderDecoder
from nlp.Seq2Seq.Mask import MaskedSoftmaxCELoss
from nlp.Seq2Seq.Seq2Seq import Seq2SeqEncoder, train_seq2seq, predict_seq2seq, bleu
from nlp.Attention.Attention import AdditiveAttention
from nlp.VocabandDataset.LoadTranslate import load_data_nmt
from nlp.gb import get_device


class AttentionDecoder(Decoder):
    """带注意力机制的Decoder接口"""""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        """
        :param vocab_size: 词典大小
        :param embed_size: 词向量维度
        :param num_hiddens: 隐藏单元个数
        :param num_layers: GRU层数
        :param dropout: dropout概率
        """""

        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # key_size, query_size, num_hiddens, dropout
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # enc_outputs = Y, state
        outputs, hidden_state = enc_outputs
        # outputs: (num_steps, batch_size, num_hiddens)
        # hidden_state: (num_layers, batch_size, num_hiddens)
        return outputs.permute(1, 0, 2), hidden_state, enc_valid_lens
        # outputs: (batch_size, num_steps, num_hiddens)

    def forward(self, X, state, *args):
        # X: (batch_size, num_steps)
        enc_outputs, hidden_state, enc_valid_lens = state
        # enc_outputs: (batch_size, num_steps, num_hiddens)
        # hidden_state: (num_layers, batch_size, num_hiddens)
        X = self.embedding(X).permute(1, 0, 2)
        # X: (num_steps, batch_size, embed_size)
        outputs, self._attention_weights = [], []
        for x in X:  # 每个时间步
            # 最开始时，hidden_state是编码器的最后一层的隐藏状态，后面每个时间步的hidden_state都是上一个时间步的输出
            # 所以使用hidden_state[-1]，即上一个时间步的输出的最后一层，unsqueeze(增加query的数量 维度)后作为query
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # query: (batch_size, 1, num_hiddens)
            # hidden_state[-1]作为query，编码器每时刻最后一层的输出作为key和value
            # 计算query与key的attention(batch_size,num_query,num_key)
            # 对value(batch_size,num_key,value_size=num_hiddens)
            # 的dim=1加权求和得到context(batch_size,1,num_hiddens)
            # 原Seq2Seq中的context是编码器每时刻最后一层的输出(num_steps,batch_size,num_hiddens)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # x是X的每个时间步的词，所以x的num_steps是1
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # x: (batch_size, num_steps=1, embed_size + num_hiddens)
            # -> (num_steps=1, batch_size, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            # out: (num_steps=1, batch_size, num_hiddens)
            # hidden_state: (num_layers, batch_size, num_hiddens)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 遍历完所有时间步后，outputs的形状是(num_steps, batch_size, num_hiddens)
        outputs = self.dense(torch.cat(outputs, dim=0))
        # outputs: (num_steps, batch_size, vocab_size)
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


# encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# encoder.eval()
# decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
# decoder.eval()
# X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
# state = decoder.init_state(encoder(X), None)
# output, state = decoder(X, state)

def test():
    device = get_device()
    print(device)
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 1

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ', f'bleu {bleu(translation, fra, k=2):.3f}')


if __name__ == '__main__':
    test()
