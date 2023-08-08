import math
from hist.hist import Hist
from hist.axis import Regular

AXIS_DICT = {
    'px': Regular(50, -200, +200, name='px', label=r'$p_{x}^{miss}$ [GeV]'),
    'py': Regular(50, -200, +200),
    'pt': Regular(50, 0, 300),
    'phi': Regular(50, -math.pi, +math.pi),
    # residual
    'residual_px': Regular(50, -200, +200),
    'residual_py': Regular(50, -200, +200),
    'residual_pt': Regular(50, -200, +200),
    'residual_phi': Regular(50, -math.pi, +math.pi),
}

def create_hist(*args) -> Hist:
    axes = [AXIS_DICT[each] for each in args]
    return Hist(*axes)
