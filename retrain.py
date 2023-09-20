#!/usr/bin/env python
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import mplhep as hep
import uproot
import uproot.writing
import yaml
from coolname import generate_slug
from torchhep.optim import configure_optimizers
from torchhep.utils.checkpoint import Checkpoint
from torchhep.utils.cuda import select_idle_gpu
from torchhep.utils.reproducibility import sample_seed
from torchhep.utils.reproducibility import set_seed
from hierconfig.config import config_field, ConfigBase
from deepmeteor.models.utils import find_model_config_cls
from deepmeteor.data.dataset import MeteorDataset
from deepmeteor.data.transformations import Standardization
from deepmeteor.data.eventweighting import EventWeightingConfig
from deepmeteor.losses.utils import find_loss_cls
from deepmeteor.result import EvaluationResult
from deepmeteor.result import EpochResult
from deepmeteor.result import Summary
from deepmeteor.learningcurve import Monitor
from deepmeteor.training import DataTransformationConfig
from deepmeteor.training import OptimizerConfig
from deepmeteor.training import train
from deepmeteor.training import evaluate
from deepmeteor.training import PhaseEnum
from deepmeteor.training import save_histogram_plots
from deepmeteor import env



@dataclass
class DataConfig(ConfigBase):
    data_dir: Optional[str] = None
    train: list[str] = config_field(default_factory=list) # FIXME
    val: list[str] = config_field(default_factory=list) # FIXME
    test: list[str] = config_field(default_factory=list) # FIXME
    batch_size: int = 256
    eval_batch_size: int = 512
    max_size: Optional[int] = 100
    pt_topk: bool = False
    entry_start: Optional[int] = None
    entry_stop: Optional[int] = None
    train_cut: Optional[str] = None
    eval_cut: Optional[str] = None

    def __post_init__(self):
        self.data_dir = self.data_dir or str(env.DATA_DIR)

    @property
    def train_files(self) -> list[str]:
        return [os.path.join(self.data_dir, each) for each in self.train] # type: ignore

    @property
    def val_files(self) -> list[str]:
        return [os.path.join(self.data_dir, each) for each in self.val] # type: ignore

    @property
    def test_files(self) -> list[str]:
        return [os.path.join(self.data_dir, each) for each in self.test] # type: ignore


@dataclass
class TrainingConfig(ConfigBase):
    loss: str = config_field(default='HuberLoss',
                             choices=('L1Loss',' MSELoss', 'HuberLoss'))
    num_epochs: int = 10
    max_grad_norm: float = 1
    num_epochs: int = 10


@dataclass
class FinetuningConfig(ConfigBase):
    src: str
    data: DataConfig = DataConfig()
    data_transformation: DataTransformationConfig = DataTransformationConfig()
    event_weighting: EventWeightingConfig = EventWeightingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()
    log_base: Optional[str] = None
    log_name: Optional[str] = None
    cuda: int = config_field(
            default=0,
            help='automatically select an idle gpu')
    seed: int = config_field(
            default=1337,
            help=('if a negative value is given, then this is set to a random '
                  'number using os.random.'))
    deterministic: bool = config_field(default=False,
                                       help='use deterministic algorithms')
    num_threads: int = 1
    sanity_check: bool = False
    batch: bool = False

    @property
    def mode(self):
        mode = 'finetuning'
        if self.sanity_check:
            mode = f'sanity-check_{mode}'
        return mode

    def __post_init__(self):
        self.log_base = self.log_base or env.LOG_DIR

        if self.cuda < 0:
            self.cuda = select_idle_gpu(as_idx=True)

        if self.log_name is None:
            now = datetime.now().strftime('%y%m%d-%H%M%S')
            slug = generate_slug(pattern=2)
            self.log_name = f'{self.mode}_{now}_{slug}' # TODO

        if self.seed < 0:
            self.seed = sample_seed()

        if self.sanity_check:
            self.data.train = self.data.train[0:1]
            self.data.val = self.data.val[0:1]
            self.data.test = self.data.test[0:1]
            self.data.entry_start = 0
            self.data.entry_stop = 4096
            self.training.num_epochs = 2

    @property
    def src_dir(self) -> Path:
        return Path(self.src)

    @property
    def log_dir(self) -> Path:
        return Path(self.log_base) / self.log_name # type: ignore


def run(config: FinetuningConfig) -> None:
    mpl.use('agg')
    hep.style.use(hep.styles.CMS)

    print(str(config))

    log_dir = config.log_dir
    log_dir.mkdir(parents=True)
    config.to_yaml(log_dir / 'config.yaml')

    src_dir = config.src_dir
    with open(src_dir / 'config.yaml') as stream:
        src_config = yaml.safe_load(stream)
    model_name = src_config['model']['name']
    model_config_cls = find_model_config_cls(model_name)
    model_config = model_config_cls.from_dict(src_config['model'])


    ###########################################################################
    # ⭐ sys
    ###########################################################################
    set_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    if config.deterministic:
        # CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(config.deterministic)
    device = torch.device(f'cuda:{config.cuda}')
    print(f'{device=}: {torch.cuda.get_device_properties(device)}')

    ###########################################################################
    # ⭐ model, loss function and optimizer
    ###########################################################################
    model = model_config.build().to(device)
    if src_config['compile']:
        model = torch.compile(model)

    checkpoint = torch.load(src_dir / 'checkpoint' / 'best_checkpoint.pt',
                            map_location=device)
    model.load_state_dict(checkpoint['model'])
    del checkpoint

    print(model)
    print(f'# of parameters = {model.num_parameters}')

    # optimizer
    optimizer = configure_optimizers(model, **asdict(config.optimizer))
    # FIXME optimizer
    print(optimizer)

    lr_scheduler = None

    # loss function
    loss_fn = find_loss_cls(config.training.loss)(reduction='none').to(device)

    ###########################################################################
    # ⭐ dataset
    ###########################################################################
    # TODO
    data_xform = Standardization.from_dict(asdict(config.data_transformation))
    event_weighting = config.event_weighting.build()

    train_set = MeteorDataset.from_root(
        path_list=config.data.train_files,
        transformation=data_xform,
        event_weighting=event_weighting,
        entry_start=config.data.entry_start,
        entry_stop=config.data.entry_stop,
        max_size=config.data.max_size,
        pt_topk=config.data.pt_topk,
    )
    print(f'{len(train_set)=}')

    val_set = MeteorDataset.from_root(
        path_list=config.data.val_files,
        transformation=data_xform,
        event_weighting=event_weighting,
        entry_start=config.data.entry_start,
        entry_stop=config.data.entry_stop,
        max_size=config.data.max_size,
        pt_topk=config.data.pt_topk,
        cut=config.data.eval_cut,
    )
    print(f'{len(val_set)=}')

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.data.batch_size,
        collate_fn=MeteorDataset.collate,
        drop_last=True,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=config.data.eval_batch_size,
        collate_fn=MeteorDataset.collate,
        drop_last=False,
        shuffle=False
    )

    ###########################################################################
    # ⭐ utils
    ###########################################################################
    checkpoint_dir = log_dir / 'checkpoint'
    checkpoint_dir.mkdir()
    checkpoint = Checkpoint(checkpoint_dir)
    checkpoint.register(
        model=model,
        optimizer=optimizer,
    )

    ###########################################################################
    # ⭐ training phase
    ###########################################################################
    monitor = Monitor()

    summary = Summary(
        num_parameters=model.num_parameters,
        step=-1,
        epoch=-1,
        # worst
        validation=EvaluationResult.worst,
        test=EvaluationResult.worst
    )

    learning_curve_dir = log_dir / 'learning_curve'
    learning_curve_dir.mkdir()

    with uproot.writing.create(learning_curve_dir / 'validation.root'):
        ...

    for epoch in range(0, 1 + config.training.num_epochs):
        print(f'\n🔥 Epoch {epoch}')

        if epoch > 0:
            train(
                model=model,
                data_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                device=device,
                config=config,
                monitor=monitor,
            )

        val_result = evaluate(
            model=model,
            data_loader=val_loader,
            data_xform=data_xform,
            loss_fn=loss_fn,
            device=device,
            phase=PhaseEnum.VALIDATION,
            output_dir=learning_curve_dir,
            config=config,
            epoch=epoch,
        )

        monitor.epoch.append(EpochResult(epoch=epoch, step=monitor.last_step))
        monitor.validation.append(val_result)
        monitor.to_csv(learning_curve_dir)
        if epoch >= 1:
            monitor.draw_all(output_dir=learning_curve_dir)

        print(str(val_result))

        checkpoint.step(loss=val_result.loss, epoch=epoch)
        if val_result.loss < summary.validation.loss:
            summary.step = monitor.last_step
            summary.epoch = epoch
            summary.validation = val_result

    ###########################################################################
    # ⭐ test phase
    ###########################################################################
    print('\n\n⭐ test phase')
    eval_dir = log_dir / 'eval'
    eval_dir.mkdir()

    test_dir = eval_dir / 'test'
    test_dir.mkdir()

    print(f'loading {checkpoint.best_path}')
    model.load_state_dict(checkpoint.best_state_dict['model'])

    test_set = MeteorDataset.from_root(
        path_list=config.data.test_files,
        transformation=data_xform,
        event_weighting=event_weighting,
        entry_start=config.data.entry_start,
        entry_stop=config.data.entry_stop,
        max_size=config.data.max_size,
        pt_topk=config.data.pt_topk,
    )

    print(f'{len(test_set)=}')
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.data.eval_batch_size,
        collate_fn=MeteorDataset.collate,
        drop_last=False,
        shuffle=False,
    )

    test_result = evaluate(
        model=model,
        data_loader=test_loader,
        data_xform=data_xform,
        loss_fn=loss_fn,
        device=device,
        output_dir=test_dir,
        phase=PhaseEnum.TEST,
        config=config,
        epoch=summary.epoch,
    )

    summary.test = test_result
    print(str(test_result))

    ###########################################################################
    # report
    ###########################################################################
    summary.to_json(log_dir / 'summary.json')
    monitor.draw_all(output_dir=learning_curve_dir, summary=summary)

    with open(log_dir / 'cuda_memory_stats.json', 'w') as stream:
        json.dump(torch.cuda.memory_stats(device), stream, indent=2)
    print(torch.cuda.memory_summary())

    hist_dir = test_dir / 'hist'
    hist_dir.mkdir()
    print('🎨 saving histogram plots')
    save_histogram_plots(
        input_path=(test_dir / 'test.root'),
        output_dir=hist_dir
    )

    ###########################################################################
    # ⭐ finish
    ###########################################################################
    print('🎉 Done!!!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path, help='Help text')
    # parser.add_argument('--src', type=Path, help='Help text')
    parser.add_argument('--sanity-check', action='store_true', help='Help text')
    parser.add_argument('--batch', action='store_true', help='Help text')
    args = parser.parse_args()

    with open(args.config) as stream:
        data = yaml.safe_load(stream)

    data['sanity_check'] = args.sanity_check
    data['batch'] = args.batch

    config = FinetuningConfig.from_dict(data)

    run(config)


if __name__ == '__main__':
    main()
