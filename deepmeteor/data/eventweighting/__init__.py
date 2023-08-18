from dataclasses import dataclass
from typing import Optional
from hierconfig.config import ConfigBase
from .noweighting import NoWeighting
from .densityweight import DensityWeightHist

EVENT_WEIGHTING_LIST = [
    NoWeighting,
    DensityWeightHist,
]

EVENT_WEIGHTING_DICT = {each.__name__: each for each in EVENT_WEIGHTING_LIST}


@dataclass
class EventWeightingConfig(ConfigBase):
    name: str = 'NoWeighting'
    file: Optional[str] = None

    def build(self):
        event_weighting_cls = EVENT_WEIGHTING_DICT[self.name]
        return event_weighting_cls.build(self.file)
