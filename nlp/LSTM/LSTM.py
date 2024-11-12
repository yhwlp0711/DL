import torch
import torch.nn as nn

from nlp.VocabandDataset.LMandDataset import load_data_time_machine
from nlp.RNN.RNNd2l import RNNModelScratch, train_ch8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    #  I = σ(W_xi*X_t + W_hi*H_t-1 + b_i)
    W_xi, W_hi, b_i = three()  # Input gate parameters
    #  F = σ(W_xf*X_t + W_hf*H_t-1 + b_f)
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    #  O = σ(W_xo*X_t + W_ho*H_t-1 + b_o)
    W_xo, W_ho, b_o = three()  # Output gate parameters
    #  C = tanh(W_xc*X_t + W_hc*H_t-1 + b_c)
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_lstm_state(batch_size, num_hiddens, device):
    #  初始化H和C
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(X@W_xi + H@W_hi + b_i)
        F = torch.sigmoid(X@W_xf + H@W_hf + b_f)
        O = torch.sigmoid(X@W_xo + H@W_ho + b_o)
        C_tilda = torch.tanh(X@W_xc + H@W_hc + b_c)
        C = F * C + I * C_tilda
        H = O * C.tanh()
        Y = H@W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H, C)


vocab_size, num_hiddens = len(vocab), 256
num_epochs, lr = 500, 1
model = RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params, init_lstm_state, lstm)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
