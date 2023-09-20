from dataclasses import dataclass
import abc
import torch
import torch.nn as nn
from torch import Tensor
from torchhep.optuna.hyperparameter import HyperparameterConfigBase
from deepmeteor.data.dataset import Batch, MeteorDataset
from deepmeteor.modules.puppiembedding import PUPPIEmbedding


@dataclass
class ModelConfigBase(HyperparameterConfigBase, metaclass=abc.ABCMeta):
    name: str
    cont_embed_dim: int = 12
    pdgid_embed_dim: int = 2
    charge_embed_dim: int = 2
    embed_dim: int = 16
    weight_dim: int = 0
    final_bias: bool = False
    final_activation: str = 'Identity'

    def __post_init__(self):
        assert (self.cont_embed_dim + self.pdgid_embed_dim + self.charge_embed_dim) == self.embed_dim
        assert 0 <= self.weight_dim <= 4, self.weight_dim
        assert hasattr(nn, self.final_activation)

    @abc.abstractmethod
    def build(self) -> 'ModelBase':
        ...

    @property
    def output_dim(self) -> int:
        """
        regression head's output dimension
        """
        if self.weight_dim == 0:
            return MeteorDataset.target_num_features
        elif self.weight_dim > 0:
            return self.weight_dim
        else:
            raise ValueError(self.weight_dim)


class ModelBase(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, config: ModelConfigBase):
        super().__init__()
        self.config = config

    @classmethod
    def build_input_embedding(cls, config: ModelConfigBase):
        return PUPPIEmbedding(
            cont_embed_dim=config.cont_embed_dim,
            pdgid_embed_dim=config.pdgid_embed_dim,
            charge_embed_dim=config.charge_embed_dim
        )

    @classmethod
    @abc.abstractmethod
    def build_encoder(cls, config: ModelConfigBase) -> nn.Module:
        ...

    @classmethod
    def build_regression_head(cls, config: ModelConfigBase):
        return nn.Sequential(
            nn.Linear(
                in_features=config.embed_dim,
                out_features=config.output_dim,
                bias=config.final_bias,
            ),
            getattr(nn, config.final_activation)()
        )

    def run(self, batch: Batch):
        output = self(
            cont=batch.puppi_cands_cont,
            pdgid=batch.puppi_cands_pdgid,
            charge=batch.puppi_cands_charge,
            data_mask=batch.puppi_cands_data_mask,
        )
        return self.reconstruct_met(output, batch.puppi_cands_pxpy)

    @property
    def num_parameters(self) -> int:
        return sum(each.numel() for each in self.parameters())

    def reconstruct_met(self,
                        input: Tensor,
                        pxpy: Tensor
    ):
        """
        """
        if self.config.weight_dim == 0:
            reco_met = input
        elif 1 <= self.config.weight_dim <= 2:
            w = 1 + input
            reco_met = -(w * pxpy).sum(dim=1)
        elif self.config.weight_dim == 3:
            dw, b_px, b_py = input.permute(2, 0, 1)
            w = dw + 1

            px, py = pxpy.permute(2, 0, 1)
            px = w * px + b_px
            py = w * py + b_py

            reco_met_px = -px.sum(dim=1)
            reco_met_py = -py.sum(dim=1)
            reco_met = torch.stack([reco_met_px, reco_met_py], dim=1)
        elif self.config.weight_dim == 4:
            dw_px, dw_py, b_px, b_py = input.permute(2, 0, 1)
            w_px = 1 + dw_px
            w_py = 1 + dw_py

            px, py = pxpy.permute(2, 0, 1)
            px = w_px * px + b_px
            py = w_py * py + b_py

            reco_met_px = -px.sum(dim=1)
            reco_met_py = -py.sum(dim=1)
            reco_met = torch.stack([reco_met_px, reco_met_py], dim=1)
        else:
            raise NotImplementedError(f'{self.config.weight_dim=}')

        return reco_met
