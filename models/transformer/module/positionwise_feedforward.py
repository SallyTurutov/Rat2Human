import torch.nn as nn
import torch.nn.functional as F

from models.transformer.encode_decode.clones import clones


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# class PositionwiseFeedForward(nn.Module):
#     "Implements FFN equation."
#     def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.0, dense_output_nonlinearity='relu'):
#         super(PositionwiseFeedForward, self).__init__()
#         self.N_dense = N_dense
#         self.linears = clones(nn.Linear(d_model, d_model), N_dense)
#         self.dropout = clones(nn.Dropout(dropout), N_dense)
#         self.leaky_relu_slope = leaky_relu_slope
#         if dense_output_nonlinearity == 'relu':
#             self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
#         elif dense_output_nonlinearity == 'tanh':
#             self.tanh = torch.nn.Tanh()
#             self.dense_output_nonlinearity = lambda x: self.tanh(x)
#         elif dense_output_nonlinearity == 'none':
#             self.dense_output_nonlinearity = lambda x: x
#
#     def forward(self, x):
#         if self.N_dense == 0:
#             return x
#
#         for i in range(len(self.linears) - 1):
#             x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))
#
#         return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))
