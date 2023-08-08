import torch
from torch import nn
from torch import Tensor
from deepmeteor import losses

def find_loss_cls(name: str):
    for module in [nn, losses]:
        if hasattr(module, name):
            loss_cls = getattr(module, name)
            if not issubclass(loss_cls, nn.Module):
                raise RuntimeError
            return loss_cls
    else:
        raise RuntimeError(f'[NOT FOUND] {name=}')


def build_loss(name_list: list[str], weight_list: list[float]):
    loss_cls_list = list(map(find_loss_cls, name_list))



def apply_loss_reduction(loss: Tensor, reduction: str) -> Tensor:
    if reduction == 'mean':
        loss = loss.mean()
    else:
        raise NotImplementedError(f'{reduction=}')

    return loss
