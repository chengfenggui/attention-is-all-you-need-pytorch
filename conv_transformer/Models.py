''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .Layers import EncoderLayer, DecoderLayer


class FeatureEmbedding(nn.Module):

    def __init__(self, in_channel=3, d_feature=32, kernel_size=5, negative_slope=0.01):
        super(FeatureEmbedding, self).__init__()

        self.negative_slope = negative_slope

        self.conv1 = nn.Conv2d(in_channel, d_feature, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv2d(in_channel, d_feature, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv3 = nn.Conv2d(in_channel, d_feature, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv4 = nn.Conv2d(in_channel, d_feature, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm2d(d_feature)
        self.bn2 = nn.BatchNorm2d(d_feature)
        self.bn3 = nn.BatchNorm2d(d_feature)
        self.bn4 = nn.BatchNorm2d(d_feature)

    def forward(self, x):
        # input size: b x c x h x w
        negative_slope = self.negative_slope

        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=negative_slope, inplace=False)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=negative_slope, inplace=False)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=negative_slope, inplace=False)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=negative_slope, inplace=False)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_feature, n_position=7):
        super(PositionalEncoding, self).__init__()

        self.n_position = n_position
        self.d_feature = d_feature

    def _get_sinusoid_encoding_table(self, n_position, d_feature, h, w):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return torch.stack([
                torch.ones([h, w]) *
                position / np.power(10000, 2 * (hid_j // 2) / d_feature)
                for hid_j in range(d_feature)
            ])

        sinusoid_table = np.array([get_position_angle_vec(pos_i).numpy() for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        n_position = self.n_position
        d_feature = self.d_feature
        b, l, c, h, w = x.size()

        pos_table = self._get_sinusoid_encoding_table(n_position, d_feature, h, w)
        return x + pos_table[:, :l].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_feature,
            d_model, d_attention, ffn_depth=2, dropout=0.1):

        super(Encoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(n_head, d_model, d_feature, d_attention, ffn_depth, dropout)
            for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm([d_model, h, w], eps=1e-6)

    def forward(self, src_seq, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward

        enc_output = self.dropout(self.position_enc(src_seq))
        # enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            # enc_slf_attn_list size: b x layer_stack x head x lq x lk x c x h x w
            return enc_output, torch.stack(enc_slf_attn_list).transpose(0, 1)
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_feature,
            d_model, d_attention, ffn_depth=2, dropout=0.1):

        super(Decoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(n_head, d_model, d_feature, d_attention, ffn_depth, dropout)
            for _ in range(n_layers)])
        # self.layer_norm = nn.LayerNorm([d_model, h, w], eps=1e-6)

    def forward(self, trg_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.dropout(self.position_enc(trg_seq))
        # dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, torch.stack(dec_slf_attn_list).transpose(0, 1), \
                   torch.stack(dec_enc_attn_list).transpose(0, 1)
        return dec_output,
