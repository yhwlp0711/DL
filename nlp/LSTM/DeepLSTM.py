import torch
from torch import nn

from nlp.RNN.RNNd2l import train_ch8
from nlp.VocabandDataset.LMandDataset import load_data_time_machine
from nlp.RNN.RNN import RNNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


def train():
    num_inputs = num_outputs = len(vocab)
    num_hiddens = 256
    num_layers = 2
    # dropout = 0.2
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = RNNModel(len(vocab), lstm_layer)
    model = model.to(device)
    num_epochs, lr = 500, 2
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)


train()
