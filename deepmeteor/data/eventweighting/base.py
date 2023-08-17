import abc
import numpy as np
import numpy.typing as npt


WeightArray = npt.NDArray[np.float32]


class EventWeighting(abc.ABC):

    @abc.abstractclassmethod
    def build(cls, file: str):
        ...

    @abc.abstractmethod
    def compute(self, input: np.ndarray) -> WeightArray:
        # NOTE only one-dim input is considered
        ...

    def __call__(self, input: np.ndarray) -> WeightArray:
        return self.compute(input)
