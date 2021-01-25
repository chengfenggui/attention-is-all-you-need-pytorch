''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_transformer.Modules import ConvolutionalAttention
from conv_transformer.UNet.Model import UNet


class MultiHeadConvAttention(nn.Module):
    ''' Multi-Head Convolutional Attention module '''

    def __init__(self, n_head, d_feature=32, d_model=32, d_attention=1, dropout=0.1):
        super(MultiHeadConvAttention, self).__init__()

        self.n_head = n_head
        self.d_feature = d_feature
        self.d_model = d_model
        self.d_attention = d_attention

        self.conv_attentions = nn.ModuleList(
            [ConvolutionalAttention(d_feature, d_model, d_attention)
             for _ in range(n_head)])
        self.fc_conv = nn.Conv2d(n_head * d_model, d_model, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # input feature maps size: b, l, c, h, w
        n_head = self.n_head
        d_model = self.d_model

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        c, h, w = q.size(2), q.size(3), q.size(4)

        multihead_q = []
        multihead_attn = []
        for head_attention in self.conv_attentions:
            output, attn = head_attention(q, k, v)
            multihead_q.append(output)
            multihead_attn.append(attn)

        # size of q: n x b x lq x c x h x w
        # size of attn: n x b x lq x lk x c x h x w
        q, attn = torch.stack(multihead_q), torch.stack(multihead_attn).transpose(0, 1)

        # Transpose to move the head dimension back: b x lq x n x c x h x w
        # Combine the last four dimensions to concatenate all the heads together: b x lq x (n*c*h*w)
        q = q.transpose(0, 1).transpose(1, 2).contiguous().view(sz_b, len_q, -1, h, w).transpose(0, 1)
        out = []
        for q_f in q:
            out.append(self.fc_conv(q_f))

        q = torch.stack(out).transpose(0, 1)
        q = self.dropout(q)

        return q, attn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    ''' A U-Net-like feed-forward network '''

    # TODO: U-Net-like structure

    def __init__(self, d_model, dropout=0.1):
        super(FeedForward, self).__init__()

        self.d_model = d_model

        self.ffn = UNet(in_channels=d_model, out_channels=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input size: b, l, c, h, w
        assert x.size(2) == self.d_model

        x = x.transpose(0, 1)
        out = []
        for fea in x:
            out.append(self.ffn(fea))
        out = torch.stack(out).transpose(0, 1)
        out = self.dropout(out)

        return out
