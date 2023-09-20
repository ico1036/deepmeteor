import numpy as np
import numpy.typing as npt
from .base import EventWeighting


class GenMETpTWeighting(EventWeighting):

    @classmethod
    def build(cls, _: str):
        return cls()

    @classmethod
    @property
    def input_name(cls):
        return None

    def compute(self,
                gen_met_pt: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        return np.copy(gen_met_pt)
