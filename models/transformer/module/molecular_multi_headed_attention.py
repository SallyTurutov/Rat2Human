import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.encode_decode.clones import clones


def attention(query, key, value, adj_matrix, distances_matrix, distance_matrix_kernel, eps=1e-6,
              lambdas=(0.3, 0.3, 0.4), mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Prepare distances matrix
    distances_matrix = distances_matrix.masked_fill(mask.repeat(1, mask.shape[-1], 1) == 0, np.inf)
    distances_matrix = distance_matrix_kernel(distances_matrix)

    if mask is not None:
        # Same mask applied to all h heads.
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    # Prepare adjacency matrix
    adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
    adj_matrix = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    p_adj = adj_matrix

    # Prepare distances matrix
    p_dist = distances_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

    lambda_attention, lambda_distance, lambda_adjacency = lambdas
    p_weighted = lambda_attention * p_attn + lambda_distance * p_dist + lambda_adjacency * p_adj

    if dropout is not None:
        p_weighted = dropout(p_weighted)

    atoms_featrues = torch.matmul(p_weighted, value)
    return atoms_featrues, p_attn


class MolecularMultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, lambda_attention=0.3, lambda_distance=0.3, distance_matrix_kernel='softmax'):
        "Take in model size and number of heads."
        super(MolecularMultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        lambda_adjacency = 1. - lambda_attention - lambda_distance
        self.lambdas = (lambda_attention, lambda_distance, lambda_adjacency)

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        if distance_matrix_kernel == 'softmax':
            self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
        elif distance_matrix_kernel == 'exp':
            self.distance_matrix_kernel = lambda x: torch.exp(-x)

    def forward(self, query, key, value, adj_matrix, distances_matrix, mask=None):
        "Implements Figure 2"

        # p_dist = distances_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

        # if mask is not None:
        #     # Same mask applied to all h heads.
        #     mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, adj_matrix, distances_matrix, self.distance_matrix_kernel,
                                 lambdas=self.lambdas, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)