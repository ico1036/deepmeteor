import abc
import numpy as np
import numpy.typing as npt


WeightArray = npt.NDArray[np.float32]


class EventWeighting(abc.ABC):

    @classmethod
    @abc.abstractmethod
    def build(cls, file: str):
        ...

    @classmethod
    @property
    @abc.abstractmethod
    def input_name(cls) -> str | None:
        ...

    @abc.abstractmethod
    def compute(self, input: np.ndarray) -> WeightArray:
        # NOTE only one-dim input is considered
        ...

    def __call__(self, input: np.ndarray) -> WeightArray:
        return self.compute(input)
