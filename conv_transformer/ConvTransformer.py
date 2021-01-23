import torch
import torch.nn as nn
from conv_transformer.Models import Encoder, Decoder, FeatureEmbedding, PositionalEncoding


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, src_in_channel, trg_in_channel, h, w, d_feature=32, n_layers=6, n_head=4,
            d_model=32, d_attention=1, embedding_kernel=5, dropout=0.1, n_position=200, slope=0.01,
            emb_src_trg_weight_sharing=True, src_embedding=False, trg_embedding=False):

        super().__init__()

        self.src_in_channel, self.trg_in_channel = src_in_channel, trg_in_channel
        self.src_embedding, self.trg_embedding = src_embedding, trg_embedding
        self.h, self.w = h, w

        if src_embedding:
            self.src_fea_embedding = FeatureEmbedding(in_channel=src_in_channel, d_feature=d_feature,
                                                      kernel_size=embedding_kernel, negative_slope=slope)
        if trg_embedding:
            self.trg_fea_embedding = FeatureEmbedding(in_channel=trg_in_channel, d_feature=d_feature,
                                                      kernel_size=embedding_kernel, negative_slope=slope)

        self.encoder = Encoder(
            n_layers=n_layers, h=h, w=w, n_head=n_head, d_feature=d_feature, d_model=d_model,
            d_attention=d_attention, dropout=dropout, n_position=n_position)

        self.decoder = Decoder(
            n_layers=n_layers, h=h, w=w, n_head=n_head, d_feature=d_feature, d_model=d_model,
            d_attention=d_attention, dropout=dropout, n_position=n_position)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if emb_src_trg_weight_sharing and src_in_channel == trg_in_channel:
            self.src_fea_embedding.weight = self.trg_fea_embedding.weight

    def forward(self, src_seq, trg_seq):
        # input size: b x l x c x h x w

        src_embedding, trg_embedding = self.src_embedding, self.trg_embedding

        len_src, len_trg, h, w = src_seq.size(1), trg_seq.size(1), src_seq.size(3), src_seq.size(4)
        assert self.h == h and self.w == w

        if src_embedding:
            src_seq = src_seq.transpose(0, 1)
            feature_maps = []
            for src_img in src_seq:
                feature_map = self.src_fea_embedding(src_img)
                feature_maps.append(feature_map)

            src_seq = torch.stack(feature_maps).transpose(0, 1)

        if trg_embedding:
            trg_seq = trg_seq.transpose(0, 1)
            feature_maps = []
            for trg_img in trg_seq:
                feature_map = self.trg_fea_embedding(trg_img)
                feature_maps.append(feature_map)

            trg_seq = torch.stack(feature_maps).transpose(0, 1)

        enc_output, *_ = self.encoder(src_seq)
        dec_output, *_ = self.decoder(trg_seq, enc_output)

        return dec_output
