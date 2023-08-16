import abc
from dataclasses import asdict
import yaml
import torch
import torch.nn as nn
from torch import Tensor
from deepmeteor.data.example import Example


class DataTransformation(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def transform_puppi_cands_cont(self, puppi_cands_cont: Tensor
    ) -> Tensor:
        ...

    @abc.abstractmethod
    def inverse_transform_puppi_cands_cont(self, puppi_cands_cont: Tensor
    ) -> Tensor:
        ...

    @abc.abstractmethod
    def transform_gen_met(self, gen_met: Tensor) -> Tensor:
        ...

    @abc.abstractmethod
    def inverse_transform_gen_met(self, gen_met: Tensor) -> Tensor:
        ...

    @abc.abstractmethod
    def transform(self, example: Example) -> Example:
        ...

    @abc.abstractmethod
    def inverse_transform(self, example: Example) -> Example:
        ...

    def __call__(self, example: Example) -> Example:
        return self.transform(example)

    @classmethod
    def from_dict(cls, data):
        data = {key: torch.tensor(value) for key, value in data.items()}
        return cls(**data)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as stream:
            data = yaml.safe_load(stream)
        return cls.from_dict(data)


class Composition(DataTransformation):
    def __init__(self, transformations: list[DataTransformation]) -> None:
        self.transformations = transformations

    def transform_puppi_cands_cont(self, puppi_cands_cont: Tensor) -> Tensor:
        for each in self.transformations:
            puppi_cands_cont = each.transform_puppi_cands_cont(
                puppi_cands_cont)
        return puppi_cands_cont

    def invesre_transform_puppi_cands_cont(self, puppi_cands_cont: Tensor
    ) -> Tensor:
        for each in reversed(self.transformations):
            puppi_cands_cont = each.inverse_transform_puppi_cands_cont(
                puppi_cands_cont)
        return puppi_cands_cont

    def transform_gen_met(self, gen_met: Tensor) -> Tensor:
        for each in self.transformations:
            gen_met = each.transform_gen_met(gen_met)
        return gen_met

    def inverse_transform_gen_met(self, gen_met: Tensor) -> Tensor:
        for each in reversed(self.transformations):
            gen_met = each.inverse_transform_gen_met(gen_met)
        return gen_met

    def transform(self, example: Example) -> Example:
        for each in self.transformations:
            example = each(example)
        return example

    def inverse_transform(self, example: Example) -> Example:
        for each in reversed(self.transformations):
            example = each(example)
        return example



class Standardization(DataTransformation):
    def __init__(self,
                 puppi_cands_cont_std: Tensor,
                 gen_met_std: Tensor,
    ) -> None:
        super().__init__()
        self.puppi_cands_cont_std = puppi_cands_cont_std
        self.gen_met_std = gen_met_std

    def transform_puppi_cands_cont(self, puppi_cands_cont: Tensor) -> Tensor:
        return puppi_cands_cont / self.puppi_cands_cont_std

    def inverse_transform_puppi_cands_cont(self, puppi_cands_cont: Tensor
    ) -> Tensor:
        return puppi_cands_cont * self.puppi_cands_cont_std

    def transform_gen_met(self, gen_met: Tensor) -> Tensor:
        return gen_met / self.gen_met_std

    def inverse_transform_gen_met(self, gen_met: Tensor) -> Tensor:
        return gen_met * self.gen_met_std

    def transform(self, example: Example) -> Example:
        fields = asdict(example)
        fields['puppi_cands_cont'] = self.transform_puppi_cands_cont(
            fields['puppi_cands_cont'])
        fields['gen_met'] = self.transform_gen_met(fields['gen_met'])
        fields['puppi_met'] = self.transform_gen_met(fields['puppi_met'])

        return Example(**fields)

    def inverse_transform(self, example: Example) -> Example:
        raise NotImplementedError

#
# def build_transformation(config) -> Composition:
#     xforms = []
#
#     for each in config:
#         if isinstance(each, str):
#             xform_cls = find_xform(each)
#         elif isinstance(each, dict):
#             xform_cls = find_xform_cls(each.pop('name'))
#
#     return xforms
