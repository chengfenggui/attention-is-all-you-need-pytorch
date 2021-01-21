''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_feature=32, d_model=32, dropout=0.1):
        super().__init__()

        self.n_head = n_head

        self.conv_qs = nn.Conv2d(d_feature, n_head * d_model, kernel_size=1)
        self.conv_ks = nn.Conv2d(d_feature, n_head * d_model, kernel_size=1)
        self.conv_vs = nn.Conv2d(d_feature, n_head * d_model, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, f_q, f_k, f_v, mask=None):
        # input feature maps size: b, lq, c, h, w
        n_head = self.n_head
        sz_b, len_q, len_k, len_v = f_q.size(0), f_q.size(1), f_k.size(1), f_v.size(1)
        c, h, w = f_q.size(2), f_q.size(3), f_q.size(4)

        residual = f_q

        # Pass through the pre-attention projection: b x lq x n x c x h x w
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(f_q).view(sz_b, len_q, n_head, c, h, w)
        k = self.w_ks(f_k).view(sz_b, len_k, n_head, c, h, w)
        v = self.w_vs(f_v).view(sz_b, len_v, n_head, c, h, w)

        # Transpose for attention dot product: b x n x lq x c x h x w
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

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
