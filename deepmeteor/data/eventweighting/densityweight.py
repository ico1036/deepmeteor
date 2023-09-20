import os
from pathlib import Path
import numpy as np
import uproot
from hist.hist import Hist
from hist.axis import Variable
from .base import EventWeighting


class DensityWeightHist(EventWeighting):
    """
    https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    bins are [
        [edges[0], edges[1]),
        ...,
        [edges[-3], edges[-2]),
        [edges[-2], edges[-1]]
    """

    def __init__(self,
                 weight_hist: np.ndarray,
                 edges: np.ndarray
    ) -> None:
        self.weight_hist = weight_hist
        self.edges = edges

    @classmethod
    def build(cls, file: str):
        return cls.from_npz(file)

    @classmethod
    @property
    def input_name(cls):
        return f'{cls.__name__}.npz'

    def compute(self, input):
        bins = np.digitize(input, bins=self.edges, right=False) - 1
        bins = np.clip(bins, 0, len(self.weight_hist) - 1)
        return self.weight_hist[bins]

    @classmethod
    def from_root(cls,
                  path_list: list[str],
                  branch: str = 'genMet_pt',
                  edges: np.ndarray = np.linspace(0, 400, 81, dtype=np.float32), # FIXME make it lazy
                  alpha: float = 1,
                  eps: float = 1e-6,
                  treepath: str = 'Events',
    ):
        """
        Args:

        Returns:
        """
        files = {each: treepath for each in path_list}
        arrays = uproot.concatenate(files=files, expressions=[branch],
                                  library='np')

        data = np.clip(arrays[branch], edges[0], edges[-1])
        count, _ = np.histogram(data, bins=edges)

        # p, frequency
        freq = count / len(data)
        # p', normalisation
        freq = (freq - freq.min()) / (freq.max() - freq.min())

        # f'
        weight = 1 - alpha * freq
        # f''
        weight = np.clip(weight, eps, None)
        # f
        avg_weight = (count * weight).sum() / count.sum()
        weight = weight / avg_weight

        return cls(weight_hist=weight, edges=edges)

    def to_hist(self) -> Hist:
        return Hist(Variable(self.edges), data=self.weight_hist) # type: ignore

    @classmethod
    def from_hist(cls, hist: Hist):
        if len(hist.axes) != 1:
            raise RuntimeError
        return cls(weight_hist=hist.values(), edges=hist.axes[0].edges)

    def to_npz(self, file) -> None:
        np.savez(file=file, weight_hist=self.weight_hist, edges=self.edges)

    @classmethod
    def from_npz(cls, file):
        npz_file = np.load(file)
        return cls(
            weight_hist=npz_file['weight_hist'],
            edges=npz_file['edges']
        )
