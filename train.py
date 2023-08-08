#!/usr/bin/env python3
import argparse
from dataclasses import field, make_dataclass
from pathlib import Path
import yaml
from deepmeteor import training
from deepmeteor.models.utils import find_model_config_cls


def make_run_config_cls(model_name: str):
    model_config_cls = find_model_config_cls(model_name)

    fields = [
        (
            'model',
            model_config_cls,
            field(default=model_config_cls())
        ),
    ]

    config_cls = make_dataclass(
        cls_name='RunConfig',
        fields=fields,
        bases=(training.RunConfigBase, )
    )
    return config_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, help='Help text')
    args = parser.parse_args()

    with open(args.config) as stream:
        data = yaml.safe_load(stream)
    model_name = data['model']['name']
    config_cls = make_run_config_cls(model_name)

    config = config_cls.from_yaml(args.config)

    training.run(config)


if __name__ == '__main__':
    main()
