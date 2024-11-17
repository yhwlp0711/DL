from torch import nn


class Encoder(nn.Module):
    """编码器接口"""""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

    def forward(self, X, *args):
        # NotImplementedError: Encoder的forward函数必须被子类实现
        raise NotImplementedError


class Decoder(nn.Module):
    """解码器接口"""""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

    def init_state(self, enc_outputs, *args):
        # NotImplementedError: Decoder的init_state函数必须被子类实现
        raise NotImplementedError

    def forward(self, X, state, *args):
        # NotImplementedError: Decoder的forward函数必须被子类实现
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码器-解码器接口"""""
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
