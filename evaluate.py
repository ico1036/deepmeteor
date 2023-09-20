#!/usr/bin/env python3
import argparse
from dataclasses import field, make_dataclass
from functools import cached_property
from pathlib import Path
import yaml
import pdfcombine
from deepmeteor.models.utils import find_model_config_cls



import os
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import mplhep as hep
from coolname import generate_slug
from torchhep.utils.cuda import select_idle_gpu
from hierconfig.config import config_field, ConfigBase
from deepmeteor.data.dataset import MeteorDataset
from deepmeteor.data.transformations import Standardization
from deepmeteor.data.eventweighting import EventWeightingConfig
from deepmeteor.models.base import ModelConfigBase
from deepmeteor.models.utils import init_weights
from deepmeteor.losses.utils import find_loss_cls
from deepmeteor import env

from deepmeteor.training import evaluate
from deepmeteor.training import PhaseEnum
from deepmeteor.training import DataTransformationConfig
from deepmeteor.training import EventWeightingConfig
from deepmeteor.training import save_histogram_plots

def combine_pdf(log_dir: Path, output_dir: Path):
    files = sorted(str(each) for each in log_dir.glob('**/*.pdf'))
    output = str(output_dir / 'plot.pdf')
    pdfcombine.combine(files=files, output=output)



@dataclass
class DataConfig(ConfigBase):
    data_dir: Optional[str] = None
    files: list[str] = config_field(default_factory=list) # FIXME
    batch_size: int = 512
    max_size: Optional[int] = 100
    pt_topk: bool = False
    entry_start: Optional[int] = None
    entry_stop: Optional[int] = None
    cut: Optional[str] = None

    def __post_init__(self):
        self.data_dir = self.data_dir or env.DATA_DIR

    @property
    def file_list(self) -> list[str]:
        return [os.path.join(self.data_dir, each) for each in self.files]


@dataclass
class EvaluationConfigBase(ConfigBase):
    model: ModelConfigBase
    src: Optional[str] = None
    name: str = 'test'
    data: DataConfig = DataConfig()
    data_transformation: DataTransformationConfig = DataTransformationConfig()
    event_weighting: EventWeightingConfig = EventWeightingConfig()
    loss: str = config_field(default='HuberLoss',
                             choices=('L1Loss',' MSELoss', 'HuberLoss'))
    cuda: int = config_field(
            default=0,
            help='automatically select an idle gpu')

    num_threads: int = 1
    sanity_check: bool = False
    batch: bool = False
    compile: bool = False

    log_base: Optional[str] = None
    log_name: Optional[str] = None


    def __post_init__(self):
        if self.cuda < 0:
            self.cuda = select_idle_gpu(as_idx=True)

        if self.sanity_check:
            self.data.files = self.data.files[0:1]
            self.data.entry_start = 0
            self.data.entry_stop = 4096

        if self.src is not None:
            self.log_base = self.log_base or env.LOG_DIR

            if self.log_name is None:
                now = datetime.now().strftime('%y%m%d-%H%M%S')
                slug = generate_slug(pattern=2)
                self.log_name = f'eval_{now}_{slug}'

    @property
    def device(self):
        if self.cuda >= 0:
            device = f'cuda:{self.cuda}'
        else:
            device = 'cpu'
        return torch.device(device)

    @property
    def has_src(self) -> bool:
        return self.src is not None

    @cached_property
    def log_dir(self) -> Path:
        if self.src is not None:
            return Path(self.src)
        else:
            return Path(self.log_base) / self.log_name

    # FIXME rename
    @classmethod
    def build(cls, model_name: str):
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
            bases=(cls, )
        )
        return config_cls






def run(config: EvaluationConfigBase) -> None:
    mpl.use('agg')
    hep.style.use(hep.styles.CMS)

    log_dir = config.log_dir
    if not config.has_src:
        log_dir.mkdir(parents=True)

    eval_dir = log_dir / 'eval' / config.name
    eval_dir.mkdir(parents=True)

    config.to_yaml(eval_dir / 'config.yaml')
    print(str(config))

    device = config.device
    torch.set_num_threads(config.num_threads)

    ###########################################################################
    # ⭐ model, loss function and optimizer
    ###########################################################################
    model = config.model.build().to(device)

    if config.has_src:
        with open(log_dir / 'config.yaml') as stream:
            src_config = yaml.safe_load(stream)
        if src_config['compile']:
            model = torch.compile(model)
        checkpoint_path = log_dir / 'checkpoint' / 'best_checkpoint.pt'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        del checkpoint
    else:
        model.apply(init_weights)

    print(model)
    print(f'# of parameters = {model.num_parameters}')

    # loss function
    loss_fn = find_loss_cls(config.loss)(reduction='none').to(device)

    ###########################################################################
    # ⭐ dataset
    ###########################################################################
    # TODO
    data_xform = Standardization.from_dict(asdict(config.data_transformation))
    event_weighting = config.event_weighting.build()

    dataset = MeteorDataset.from_root(
        path_list=config.data.file_list,
        transformation=data_xform,
        event_weighting=event_weighting,
        entry_start=config.data.entry_start,
        entry_stop=config.data.entry_stop,
        max_size=config.data.max_size,
        pt_topk=config.data.pt_topk,
        cut=config.data.cut,
    )

    print(f'{len(dataset)=}')
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.data.batch_size,
        collate_fn=MeteorDataset.collate,
        drop_last=False,
        shuffle=False,
    )

    result = evaluate(
        model=model,
        data_loader=data_loader,
        data_xform=data_xform,
        loss_fn=loss_fn,
        device=device,
        output_dir=eval_dir,
        phase=PhaseEnum.TEST,
        config=config,
        epoch=-1,
    )

    hist_dir = eval_dir / 'hist'
    hist_dir.mkdir()
    print('🎨 saving histogram plots')
    save_histogram_plots(
        input_path=(eval_dir / 'test.root'),
        output_dir=hist_dir
    )

    ###########################################################################
    # ⭐ finish
    ###########################################################################
    print('🎉 Done!!!')

    combine_pdf(config.log_dir, output_dir=eval_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, help='Help text')
    parser.add_argument('--sanity-check', action='store_true', help='Help text')
    parser.add_argument('--batch', action='store_true', help='Help text')
    args = parser.parse_args()

    with open(args.config) as stream:
        data = yaml.safe_load(stream)
    data['sanity_check'] |= args.sanity_check
    data['batch'] |= args.batch

    config_cls = EvaluationConfigBase.build(data['model']['name'])
    config = config_cls.from_dict(data)
    run(config)



if __name__ == '__main__':
    main()
