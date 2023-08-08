import importlib
import torch.nn as nn


def init_weights(module):
    """adapted from https://github.com/karpathy/nanoGPT/blob/master/model.py#L162-L168"""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


def find_model_config_cls(name):
    model_module = importlib.import_module(f'deepmeteor.models.{name.lower()}')
    return getattr(model_module, f'{name}Config')
