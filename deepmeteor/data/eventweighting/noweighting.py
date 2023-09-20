import numpy as np
from .base import EventWeighting


class NoWeighting(EventWeighting):

    @classmethod
    def build(cls, _: str):
        return cls()

    @classmethod
    @property
    def input_name(cls):
        return None

    def compute(self, input):
        return np.ones(len(input), dtype=np.float32)
