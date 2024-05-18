import torch
import torch.nn as nn
import numpy as np
import math

from models.dataset_utils import LongTensor
import configuration.config_default as cfgd


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # weight matrix, each row present one word
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    # def forward(self, x, tissues_list):
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
