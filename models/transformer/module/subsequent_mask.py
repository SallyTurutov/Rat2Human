import numpy as np
import torch


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)  # +1 due to the addition of tissue in embedding
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
