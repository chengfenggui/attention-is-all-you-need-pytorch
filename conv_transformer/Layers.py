''' Define the Layers '''
import torch.nn as nn
import torch
from .SubLayers import MultiHeadConvAttention, Residual, FeedForward


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, n_head=4, d_model=32, d_feature=32, d_attention=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadConvAttention(n_head, d_feature, d_model, d_attention, dropout=dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, enc_input):
        residual = enc_input
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output += residual

        residual = enc_output
        enc_output = self.ffn(enc_output)
        enc_output += residual
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, n_head=4, d_model=32, d_feature=32, d_attention=1, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadConvAttention(n_head, d_feature, d_model, d_attention, dropout=dropout)
        self.enc_attn = MultiHeadConvAttention(n_head, d_feature, d_model, d_attention, dropout=dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(
            self, dec_input, enc_output):
        residual = dec_input

        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input)
        dec_output += residual

        residual = dec_output

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output)
        dec_output += residual

        residual = dec_output

        dec_output = self.ffn(dec_output)
        dec_output += residual

        return dec_output, dec_slf_attn, dec_enc_attn
