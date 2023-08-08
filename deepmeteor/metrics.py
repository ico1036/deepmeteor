from dataclasses import dataclass
import numpy as np
import scipy as sp
from hist.hist import Hist


@dataclass
class Hist1DStat:
    mean: float
    std: float

    @classmethod
    def from_hist(cls, histogram: Hist):
        x_axis = histogram.axes[0]

        probs = histogram.density() * x_axis.widths
        mean = np.sum(probs * x_axis.centers)
        std = np.sqrt(np.sum(probs * (x_axis.centers - mean)**2))
        return cls(mean.item(), std.item())

    def __str__(self):
        return f'µ={self.mean:.3f}, σ={self.std:.3f}'

def compute_reduced_chi2(h1: np.ndarray | Hist,
                 h2: np.ndarray | Hist
):
    """only for UU
    """
    def to_numpy(histogram) -> np.ndarray:
        if isinstance(histogram, Hist):
            histogram, _ = histogram.to_numpy()
        return histogram

    h1 = to_numpy(h1)
    h2 = to_numpy(h2)

    if h2.shape != h2.shape:
        raise RuntimeError

    sum1 = h1.sum()
    sum2 = h2.sum()

    mask = (h1 != 0) | (h2 != 0)
    h1 = h1[mask]
    h2 = h2[mask]
    ndf = mask.sum() - 1

    delta_arr = sum2 * h1 - sum1 * h2
    chi2_arr = delta_arr ** 2 / (h1 + h2)
    chi2 = chi2_arr.sum() / (sum1 * sum2)

    #return 1 - sp.stats.chi2.cdf(chi2, ndf)
    return chi2 / ndf
