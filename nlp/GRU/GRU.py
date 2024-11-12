import torch
from torch import nn

from nlp.VocabandDataset.LMandDataset import load_data_time_machine
from nlp.RNN.RNNd2l import RNNModelScratch, train_ch8
from nlp.RNN.RNN import RNNModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
num_epochs, lr = 500, 0.1

num_inputs = num_outputs = len(vocab)
num_hiddens = 256
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = RNNModel(len(vocab), gru_layer)
model = model.to(device)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
