#!/usr/bin/env python3
import argparse
from dataclasses import field, make_dataclass
from pathlib import Path
import yaml
import pdfcombine
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

def combine_pdf(log_dir: Path):
    files = sorted(str(each) for each in log_dir.glob('**/*.pdf'))
    output = str(log_dir / 'plot.pdf')
    pdfcombine.combine(files=files, output=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, help='Help text')
    parser.add_argument('--sanity-check', action='store_true', help='Help text')
    parser.add_argument('--batch', action='store_true', help='Help text')
    args = parser.parse_args()

    with open(args.config) as stream:
        data = yaml.safe_load(stream)
    model_name = data['model']['name']
    config_cls = make_run_config_cls(model_name)

    data['sanity_check'] |= args.sanity_check
    data['batch'] |= args.batch

    config = config_cls.from_dict(data)
    training.run(config)

    combine_pdf(config.log_dir)


if __name__ == '__main__':
    main()
