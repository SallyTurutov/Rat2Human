import torch
import torch.nn as nn
import torch.nn.functional as F

import configuration.config_default as cfgd


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab_size, use_generator_tissue_specification):
        super(Generator, self).__init__()
        self.use_generator_tissue_specification = use_generator_tissue_specification

        if self.use_generator_tissue_specification:
            self.d_model = d_model
            self.vocab_size = vocab_size
            self.N_tissues = len(cfgd.TISSUES)

            # Define linear projection matrix for each domain
            self.proj = nn.Linear(d_model // self.N_tissues * self.N_tissues, vocab_size)

            # Define linear projection matrix for down-projection
            self.down_projection = nn.Linear(d_model, d_model // self.N_tissues)
        else:
            self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x, tissues_list):
        """
        x: decoder state tensor of shape (batch_size, seq_len, d_model)
        tissues_list: domain indicator tensor of shape (batch_size,) containing domain index for each example in the batch
        """
        if self.use_generator_tissue_specification:
            # Apply down-projection
            down_projected_x = self.down_projection(x)

            # Initialize an empty list to store tensors for concatenation
            concatenated_tensors = []

            # Iterate over batch examples
            for i in range(x.size(0)):
                # Determine the index of the zero tensor
                zero_index = tissues_list[i]
                tensors = []

                # Concatenate tensors up to zero_index
                for _ in range(zero_index):
                    tensors.append(down_projected_x[i])

                # Concatenate zero tensor
                tensors.append(torch.zeros_like(down_projected_x[i]))

                # Concatenate the rest of the tensors
                for _ in range(zero_index + 1, self.N_tissues):
                    tensors.append(down_projected_x[i])

                # Concatenate tensors and append to the list
                concatenated_tensors.append(torch.cat(tensors, dim=-1))

            # Stack the concatenated tensors into a single tensor
            concatenated_tensors = torch.stack(concatenated_tensors)

            # Apply softmax operation
            output = F.log_softmax(self.proj(concatenated_tensors), dim=-1)

            return output
        else:
            return F.log_softmax(self.proj(x), dim=-1)
