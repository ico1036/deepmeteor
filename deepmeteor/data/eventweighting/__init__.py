from dataclasses import dataclass
from typing import Optional
from hierconfig.config import ConfigBase
from deepmeteor.env import find_from_data_dir
from .noweighting import NoWeighting
from .densityweight import DensityWeightHist
from .genmetptweight import GenMETpTWeighting
from .genmetptweightxdensityweight import GenMETpTAndDensityWeightHist


EVENT_WEIGHTING_LIST = [
    NoWeighting,
    DensityWeightHist,
    GenMETpTWeighting,
    GenMETpTAndDensityWeightHist,
]


EVENT_WEIGHTING_DICT = {each.__name__: each for each in EVENT_WEIGHTING_LIST}


@dataclass
class EventWeightingConfig(ConfigBase):
    name: str = 'NoWeighting'
    file: Optional[str] = None

    def __post_init__(self) -> None:
        if self.file is None:
            weighting_cls = EVENT_WEIGHTING_DICT[self.name]
            if weighting_cls.input_name is not None:
                self.file = str(find_from_data_dir(weighting_cls.input_name))

    def build(self):
        event_weighting_cls = EVENT_WEIGHTING_DICT[self.name]
        return event_weighting_cls.build(self.file)
