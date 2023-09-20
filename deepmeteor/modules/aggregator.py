from torch import Tensor
import torch.nn as nn



def masked_mean(
            input: Tensor,
            data_mask: Tensor,
) -> Tensor:
    """
    Args:
        input: a tensor with the shape of (N, L, E)
        data_mask: a boolean tensor with the shape of (N, L)
    Returns:
        mean: a tensor with the shape of (N, E)
    """
    pad_mask = data_mask.logical_not().unsqueeze(-1)
    input = input.masked_fill(pad_mask, 0)
    return input.mean(dim=1)


class MaskedMean(nn.Module):

    def forward(self,
                input: Tensor,
                data_mask: Tensor,
    ) -> Tensor:
        return masked_mean(input, data_mask)
