import torch.nn as nn

from models.transformer.encode_decode.clones import clones
from models.transformer.encode_decode.sublayer_connection import SublayerConnection


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout, use_encoder_mol_attention):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.use_encoder_mol_attention = use_encoder_mol_attention

    def forward(self, x, mask, adj_matrix, distances_matrix):
        "Follow Figure 1 (left) for connections."
        if self.use_encoder_mol_attention:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, adj_matrix, distances_matrix, mask))
            return self.sublayer[1](x, self.feed_forward)
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
            return self.sublayer[1](x, self.feed_forward)
