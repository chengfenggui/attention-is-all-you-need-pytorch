import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalAttention(nn.Module):
    ''' Convolutional Attention '''

    def __init__(self, d_feature=32, d_model=32, d_attention=1, attn_dropout=0.1):
        super(ConvolutionalAttention, self).__init__()
        self.d_feature = d_feature
        self.d_model = d_model
        self.d_attention = d_attention

        self.conv_qs = nn.Conv2d(d_feature, d_model, kernel_size=1)
        self.conv_ks = nn.Conv2d(d_feature, d_model, kernel_size=1)
        self.conv_vs = nn.Conv2d(d_feature, d_model, kernel_size=1)
        self.conv_attenion = nn.Conv2d(d_model, d_attention, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        # input size of: b x l x c x h x w
        d_feature = self.d_feature
        d_model = self.d_model

        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        c, h, w = q.size(2), q.size(3), q.size(4)
        assert len_k == len_v and c == d_feature

        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        # Pass through the pre-attention projection: l x b x c x h x w
        q = torch.stack([self.conv_qs(x) for x in q]).view(len_q, sz_b, d_model, h, w)
        k = torch.stack([self.conv_ks(x) for x in k]).view(len_k, sz_b, d_model, h, w)
        v = torch.stack([self.conv_vs(x) for x in v]).view(len_v, sz_b, d_model, h, w)

        output = []
        attentions = []
        for i in len_q:
            qk = torch.stack([q[i, ...]] * len_k) + k
            attn = torch.stack([self.conv_attenion(x) for x in qk])
            attn = F.softmax(attn, dim=0)
            attn = self.dropout(attn)

            res = torch.mul(attn, v).sum(dim=0)
            output.append(res)
            attentions.append(attn.transpose(0, 1))

        output = torch.stack(output).transpose(0, 1)
        attentions = torch.stack(attentions).transpose(0, 1)

        # output size: b x lq x c x h x w
        # attentions size: b x lq x lk x c x h x w
        return output, attentions
