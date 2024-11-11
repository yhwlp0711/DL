import torch
import torch.nn as nn

from NLP.nlp.gb import get_device
from NLP.nlp.VocabandDataset.LMandDataset import load_data_time_machine
from NLP.nlp.RNN.RNNd2l import RNNModelScratch, train_ch8

device = get_device()
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        # XH, HH, B
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # Z = (X@W_xz) + (H@W_hz) + B_z
    W_xz, W_hz, b_z = three()  # Update gate parameters
    # R = (X@W_xr) + (H@W_hr) + B_r
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    # H_tilda = (X@W_xh) + ((R*H)@W_hh) + B_h
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    # H = (1-Z)*H + (Z*H_tilda)
    # Y = H*W_hq + B_q
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid(X@W_xz + H@W_hz + b_z)
        R = torch.sigmoid(X@W_xr + H@W_hr + b_r)
        H_tilda = torch.tanh(X@W_xh + ((R*H)@W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


vocab_size, num_hiddens = len(vocab), 256
num_epochs, lr = 500, 1
model = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_gru_state, gru)
train_ch8(model, train_iter, vocab, lr, num_epochs, device)
