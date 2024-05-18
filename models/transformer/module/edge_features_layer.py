import torch
import torch.nn as nn


class EdgeFeaturesLayer(nn.Module):
    def __init__(self, d_model, d_edge, h):
        super(EdgeFeaturesLayer, self).__init__()
        assert d_model % h == 0
        self.linear = nn.Linear(d_edge, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.25)

    def forward(self, x):
        p_edge = x.permute(0, 2, 3, 1)
        p_edge = self.linear(p_edge).permute(0, 3, 1, 2)
        return torch.relu(p_edge)