from typing import Sequence
from torch import Tensor
import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, dims: Sequence[int]) -> None:
        super().__init__()

        self.dims = dims

    def forward(self, input: Tensor):
        return input.permute(*self.dims).contiguous()

    def __repr__(self) -> str:
        name = self.__class__.__name__
        dims = self.dims
        return f'{name}({dims=})'
