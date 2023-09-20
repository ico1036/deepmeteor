"""
"""
from dataclasses import dataclass
import torch.nn as nn
from torch import Tensor
from deepmeteor.models.base import ModelBase
from deepmeteor.models.base import ModelConfigBase
from deepmeteor.data.dataset import Batch
from deepmeteor.modules.reshape import Permute
from deepmeteor.modules.aggregator import masked_mean

@dataclass
class DeepMETConfig(ModelConfigBase):
    name: str = 'DeepMET' # TODO note
    activation: str = 'ReLU'
    num_layers: int = 1

    def build(self):
        return DeepMET(self)


class DeepMET(ModelBase):
    config: DeepMETConfig

    def __init__(self, config: DeepMETConfig):
        super().__init__(config)

        self.input_embedding = self.build_input_embedding(config)
        self.encoder = self.build_encoder(config)
        self.regression_head = self.build_regression_head(config)


    @classmethod
    def build_encoder(cls, config: DeepMETConfig):
        layers: list[nn.Module] = []
        for _ in range(config.num_layers):
            layers += [
                nn.Linear(config.embed_dim, config.embed_dim),
                getattr(nn, config.activation)(),
                Permute([0, 2, 1]), # (N, L, C) -> (N, C, L)
                nn.BatchNorm1d(config.embed_dim),
                Permute([0, 2, 1]), # (N, C, L) -> (N, L, C)
            ]
        return nn.Sequential(*layers)

    def forward(self,
                cont: Tensor,
                pdgid: Tensor,
                charge: Tensor,
                data_mask: Tensor,
    ) -> Tensor:
        """
        """
        z = self.input_embedding(cont=cont, pdgid=pdgid, charge=charge)
        z = self.encoder(z)
        if self.config.weight_dim == 0:
            z = masked_mean(z, data_mask)
        return self.regression_head(z)
