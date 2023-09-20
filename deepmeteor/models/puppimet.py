from dataclasses import dataclass
from torch import Tensor
from deepmeteor.models.base import ModelBase
from deepmeteor.models.base import ModelConfigBase
from deepmeteor.data.dataset import Batch

@dataclass
class PuppiMetConfig(ModelConfigBase):
    name: str = 'PuppiMet' # TODO note

    def build(self):
        return PuppiMet(self)

class PuppiMet(ModelBase):

    def __init__(self, config: PuppiMetConfig):
        super().__init__(config)

    def forward(self,
                puppi_cands_pxpy: Tensor,
    ) -> Tensor:
        reco_met = -puppi_cands_pxpy.sum(dim=1)
        return reco_met

    # TODO rename
    def run(self, batch: Batch):
        return self(
            puppi_cands_pxpy=batch.puppi_cands_pxpy,
        )
