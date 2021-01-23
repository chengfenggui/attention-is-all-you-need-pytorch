''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from conv_transformer.Modules import ConvolutionalAttention


class MultiHeadConvAttention(nn.Module):
    ''' Multi-Head Convolutional Attention module '''

    def __init__(self, n_head, h, w, d_feature=32, d_model=32, d_attention=1, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_feature = d_feature
        self.d_model = d_model
        self.d_attention = d_attention
        self.h = h
        self.w = w

        self.conv_attentions = nn.ModuleList(
            [ConvolutionalAttention(d_feature, d_model, d_attention)
             for _ in range(n_head)])
        self.fc = nn.Linear(n_head * d_model * h * w, d_model * h * w, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # input feature maps size: b, l, c, h, w
        n_head = self.n_head
        d_model = self.d_model

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        c, h, w = q.size(2), q.size(3), q.size(4)
        assert self.h == h and self.w == w

        multihead_q = []
        multihead_attn = []
        for head_attention in self.conv_attentions:
            output, attn = head_attention(q, k, v)
            multihead_q.append(output)
            multihead_attn.append(attn)

        # size of q: n x b x lq x c x h x w
        # size of attn: n x b x lq x lk x c x h x w
        q, attn = torch.stack(multihead_q), torch.stack(multihead_attn)

        # Transpose to move the head dimension back: b x lq x n x c x h x w
        # Combine the last four dimensions to concatenate all the heads together: b x lq x (n*c*h*w)
        q = q.transpose(0, 1).transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q = q.view(sz_b, len_q, d_model, h, w)
        q = self.dropout(q)

        return q, attn


class NormResidual(nn.Module):
    def __init__(self, d_model, h, w):
        super().__init__()
        self.layer_norm = nn.LayerNorm([d_model, h, w], eps=1e-6)

    def forward(self, x, q):
        out = q + x
        out = self.layer_norm(out)

        return out


class FeedForward(nn.Module):
    ''' A U-Net-like feed-forward network '''

    # TODO: U-Net-like structure

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
