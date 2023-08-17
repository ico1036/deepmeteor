import numpy as np
from .base import EventWeighting


class NoWeighting(EventWeighting):

    @classmethod
    def build(cls, _: str):
        return cls()

    def compute(self, input):
        return np.ones(len(input), dtype=np.float32)
