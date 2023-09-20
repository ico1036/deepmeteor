from dataclasses import dataclass
from dataclasses import asdict
from dataclasses import fields
import json


@dataclass
class ResultBase:

    def __str__(self):
        return json.dumps(asdict(self), indent=4)

    def to_json(self, path):
        with open(path, 'w') as stream:
            json.dump(asdict(self), stream, indent=2)

    def __getitem__(self, key):
        return getattr(self, key)

    @classmethod
    @property
    def field_names(cls):
        return [each.name for each in fields(cls)]

@dataclass
class TrainingResult(ResultBase):
    loss: float


@dataclass
class EvaluationResult(ResultBase):
    loss: float
    loss_0_30: float
    loss_30_60: float
    loss_60_100: float
    loss_100_150: float
    loss_150_inf: float
    # reduced chi2 = chi2 / ndf
    reduced_chi2_px: float
    reduced_chi2_py: float
    reduced_chi2_pt: float
    reduced_chi2_phi: float
    # relative to puppi
    reduced_chi2_px_ratio: float
    reduced_chi2_py_ratio: float
    reduced_chi2_pt_ratio: float
    reduced_chi2_phi_ratio: float
    # mean and std dev of residuals
    residual_px_mean: float
    residual_py_mean: float
    residual_pt_mean: float
    residual_phi_mean: float
    ## std dev
    residual_px_std: float
    residual_py_std: float
    residual_pt_std: float
    residual_phi_std: float
    ## relative to puppi
    residual_px_std_ratio: float
    residual_py_std_ratio: float
    residual_pt_std_ratio: float
    residual_phi_std_ratio: float

    @classmethod
    @property
    def worst(cls):
        args = [float('inf')] * len(fields(cls))
        return cls(*args)

# silly one...
@dataclass
class EpochResult(ResultBase):
    epoch: int
    step: int


@dataclass
class Summary(ResultBase):
    num_parameters: int
    # best
    step: int
    epoch: int
    validation: EvaluationResult
    test: EvaluationResult
