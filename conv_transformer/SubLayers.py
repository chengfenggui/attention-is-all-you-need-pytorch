''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .UNet.Model import UNet
from .Modules import SEResblock
from .UNet.Parts import DoubleConv


class MultiHeadConvAttention(nn.Module):
    ''' Multi-Head Convolutional Attention module '''

    def __init__(self, n_head, d_feature=32, d_model=32, d_attention=1, dropout=0.1):
        super(MultiHeadConvAttention, self).__init__()

        self.n_head = n_head
        self.d_feature = d_feature
        self.d_model = d_model
        self.d_attention = d_attention

        self.conv_qs = nn.Conv2d(d_feature, d_model * n_head, kernel_size=1)
        self.conv_ks = nn.Conv2d(d_feature, d_model * n_head, kernel_size=1)
        self.conv_vs = nn.Conv2d(d_feature, d_model * n_head, kernel_size=1)
        self.conv_attenion = nn.Conv2d(d_model, d_attention, kernel_size=5, padding=2)

        self.fc = nn.Linear(n_head * d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, q, k, v):
        # input feature maps size: b, l, c, h, w
        n_head = self.n_head
        d_model = self.d_model
        d_feature = self.d_feature
        d_attention = self.d_attention

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        c, h, w = q.size(2), q.size(3), q.size(4)
        assert len_k == len_v and c == d_feature

        # Pass through the pre-attention projection: l x b x c x h x w
        q = self.lrelu(self.conv_qs(q.view(sz_b * len_q, c, h, w))).view(sz_b, len_q, n_head, c, h, w)
        k = self.lrelu(self.conv_ks(k.view(sz_b * len_k, c, h, w))).view(sz_b, len_k, n_head, c, h, w)
        v = self.lrelu(self.conv_vs(v.view(sz_b * len_v, c, h, w))).view(sz_b, len_v, n_head, c, h, w)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        qs = []
        attns = []

        for i in range(len_q):
            qi = q[:, :, i, :, :, :].view(sz_b, n_head, 1, c, h, w).expand_as(k)
            qk = qi + k
            attn = self.conv_attenion(qk.view(sz_b * n_head * len_k, c, h, w))
            attn = attn.view(sz_b, n_head, len_k, d_attention, h, w)
            attn = F.softmax(attn, dim=2)

            res = torch.mul(attn, v).sum(dim=2)

            qs.append(res)
            attns.append(attn)

        # size of qs: b x n x lq x c x h x w
        # size of attns: n x b x lq x lk x c x h x w
        qs = torch.stack(qs, dim=2)
        attns = torch.stack(attns, dim=2)

        # Transpose to move the head dimension back: b x lq x n x c x h x w
        # Combine the two dimensions to concatenate all the heads together: b x lq x (n*c) x h x w
        q = qs.transpose(1, 2).contiguous().view(sz_b, len_q, -1, h, w)
        q = self.fc(q.permute(0, 1, 3, 4, 2))

        q = self.dropout(q.permute(0, 1, 4, 2, 3))

        return q, attns


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    ''' A U-Net-like feed-forward network '''

    def __init__(self, d_model, ffn_depth=2, dropout=0.1):
        super(FeedForward, self).__init__()

        self.d_model = d_model

        # self.ffn = UNet(in_channels=d_model, out_channels=d_model, depth=ffn_depth)
        # self.ffn = DoubleConv(in_channels=d_model, out_channels=d_model, mid_channels=2 * d_model)
        self.convs = nn.Sequential(
            SEResblock(in_channel=d_model, out_channel=d_model, mid_channel=2 * d_model),
            SEResblock(in_channel=d_model, out_channel=d_model, mid_channel=2 * d_model),
            SEResblock(in_channel=d_model, out_channel=d_model, mid_channel=2 * d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input size: b, l, c, h, w
        b, l, c, h, w = x.shape
        assert c == self.d_model

        x = x.view(b * l, c, h, w)
        # x = self.ffn(x)
        x = self.convs(x)
        x = self.dropout(x.view(b, l, c, h, w))

        return x
