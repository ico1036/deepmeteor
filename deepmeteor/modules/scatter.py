import torch
import torch.nn as nn
from torch import Tensor


def scatter_add(input: Tensor, data_mask: Tensor, length: Tensor) -> Tensor:
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
    return output


class ScatterAdd(nn.Module):
    def forward(self,
                input: Tensor,
                data_mask: Tensor,
                length: Tensor
    ) -> Tensor:
        """scatter mean
        """
        return scatter_add(input=input, data_mask=data_mask, length=length)


def scatter_mean(input: Tensor, data_mask: Tensor, length: Tensor) -> Tensor:
    output = scatter_add(input=input, data_mask=data_mask, length=length)
    output = output / length.unsqueeze(1).to(output.dtype)
    return output


class ScatterMean(nn.Module):
    def forward(self,
                input: Tensor,
                data_mask: Tensor,
                length: Tensor
    ) -> Tensor:
        """scatter mean
        """
        return scatter_mean(input=input, data_mask=data_mask, length=length)
