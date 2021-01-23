''' Define the Layers '''
import torch.nn as nn
import torch
from conv_transformer.SubLayers import MultiHeadConvAttention, NormResidual, FeedForward


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, h, w, n_head=4, d_model=32, d_feature=32, d_attention=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadConvAttention(n_head, h, w, d_feature, d_model, d_attention, dropout=dropout)
        self.add1 = NormResidual(d_model, h, w)
        self.ffn = FeedForward(1, 1, dropout=dropout)
        self.add2 = NormResidual(d_model, h, w)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.add1(enc_output)
        enc_output = self.ffn(enc_output)
        enc_output = self.add2(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, h, w, n_head=4, d_model=32, d_feature=32, d_attention=1, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadConvAttention(n_head, h, w, d_feature, d_model, d_attention, dropout=dropout)
        self.add1 = NormResidual(d_model, h, w)
        self.enc_attn = MultiHeadConvAttention(n_head, h, w, d_feature, d_model, d_attention, dropout=dropout)
        self.add2 = NormResidual(d_model, h, w)
        self.ffn = FeedForward(1, 1, dropout=dropout)
        self.add3 = NormResidual(d_model, h, w)

    def forward(
            self, dec_input, enc_output):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input)
        dec_output = self.add1(dec_output)

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output)
        dec_output = self.add2(dec_output)

        dec_output = self.ffn(dec_output)
        dec_output = self.add3(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn
