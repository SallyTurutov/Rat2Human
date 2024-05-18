import torch.nn as nn

from models.transformer.encode_decode.clones import clones
from models.transformer.encode_decode.layer_norm import LayerNorm


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, trg_mask):
        m = memory
        for layer in self.layers:
            x = layer(x, m, src_mask, trg_mask)
        return self.norm(x)
