import numpy as np
from .densityweight import DensityWeightHist

# silly thing... plan to develop `Compose`
class GenMETpTAndDensityWeightHist(DensityWeightHist):

    @classmethod
    @property
    def input_name(cls):
        return f'DensityWeightHist.npz'

    def compute(self, gen_met_pt):
        bins = np.digitize(gen_met_pt, bins=self.edges, right=False) - 1
        bins = np.clip(bins, 0, len(self.weight_hist) - 1)
        return gen_met_pt * self.weight_hist[bins]
