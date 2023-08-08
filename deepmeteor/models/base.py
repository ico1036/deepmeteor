from dataclasses import dataclass
import abc
import torch.nn as nn
from torch import Tensor
from hierconfig.config import ConfigBase
from deepmeteor.data.dataset import Batch


@dataclass
class ModelConfigBase(ConfigBase, metaclass=abc.ABCMeta):
    name: str

    @abc.abstractmethod
    def build(self) -> 'ModelBase':
        ...


class ModelBase(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self, batch: Batch) -> Tensor:
        ...

    @property
    def num_parameters(self) -> int:
        return sum(each.numel() for each in self.parameters())
