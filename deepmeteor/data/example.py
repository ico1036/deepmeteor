from dataclasses import dataclass
from torch import Tensor
from torchhep.data.utils import TensorCollection

@dataclass
class Example(TensorCollection):
    puppi_cands_cont: Tensor # all continuous variables: px, py, eta and PUPPI
    puppi_cands_pdgid: Tensor # categorical data
    puppi_cands_charge: Tensor # categorical
    gen_met: Tensor
    puppi_met: Tensor
    weight: Tensor
    gen_met_pt: Tensor
    puppi_cands_pxpy: Tensor

    @property
    def target(self) -> Tensor:
        return self.gen_met


@dataclass
class Batch(TensorCollection):
    puppi_cands_cont: Tensor
    puppi_cands_pdgid: Tensor
    puppi_cands_charge: Tensor
    puppi_cands_data_mask: Tensor
    puppi_cands_length: Tensor
    gen_met: Tensor
    puppi_met: Tensor
    weight: Tensor
    gen_met_pt: Tensor
    puppi_cands_pxpy: Tensor

    def __len__(self):
        return len(self.puppi_cands_cont)

    @property
    def target(self) -> Tensor:
        return self.gen_met
