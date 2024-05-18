import torch.nn as nn

from models.transformer.encode_decode.clones import clones
from models.transformer.encode_decode.sublayer_connection import SublayerConnection


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    # def forward(self, x, memory, src_mask, trg_mask):
    def forward(self, x, memory, src_mask, trg_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, trg_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

    # def forward(self, x, memory, src_mask, trg_mask, src_adj_matrix, src_distances_matrix, trg_adj_matrix, trg_distances_matrix):
    #     "Follow Figure 1 (right) for connections."
    #     m = memory
    #     x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, trg_adj_matrix, trg_distances_matrix, trg_mask))
    #     x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_adj_matrix, src_distances_matrix, src_mask))
    #     return self.sublayer[2](x, self.feed_forward)
