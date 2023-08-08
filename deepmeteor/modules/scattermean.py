import torch
from torch import Tensor
import torch.nn as nn


class ScatterMean(nn.Module):
    def forward(self,
                input: Tensor,
                data_mask: Tensor,
                length: Tensor
    ) -> Tensor:
        """scatter mean
        """
        batch_size = input.size(0)
        input_dim = input.size(2)

        data_mask = data_mask.unsqueeze(2)
        input = input.masked_select(data_mask)
        input = input.reshape(-1, input_dim)

        index = torch.arange(length.size(0), device=input.device)
        index = index.repeat_interleave(length, dim=0)
        index = index.unsqueeze(1).repeat(1, input_dim)

        output = input.new_zeros((batch_size, input_dim))
        output = output.scatter_add(dim=0, index=index, src=input)
        output = output / length.unsqueeze(1).to(output.dtype)
        return output
