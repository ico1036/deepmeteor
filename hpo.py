#!/usr/bin/env python
import os
from pathlib import Path
from typing import Any, Optional, Type
from dataclasses import dataclass, asdict
from datetime import datetime
from deepmeteor.models.base import ModelConfigBase
import torch
from torch import nn
from torch.utils.data import DataLoader
from optuna import Trial

from coolname import generate_slug

from hierconfig.config import ConfigBase, config_field
from torchhep.optuna.objective import ObjectiveBase
from torchhep.optuna.study import run_study
from torchhep.optim import configure_optimizers
from torchhep.utils.cuda import select_idle_gpu
import torchhep.optuna.hyperparameter as hp

from deepmeteor import training

from deepmeteor.data.dataset import MeteorDataset
from deepmeteor.data.transformations import DataTransformation
from deepmeteor.data.transformations import Standardization
from deepmeteor.data.eventweighting import EventWeightingConfig
from deepmeteor.losses.utils import find_loss_cls
from deepmeteor.models.utils import find_model_config_cls
from deepmeteor.training import DataTransformationConfig
from deepmeteor.training import OptimizerConfig


@dataclass
class DataConfig(ConfigBase):
    data_dir: Optional[str] = None
    train: list[str] = config_field(default_factory=list) # FIXME
    val: list[str] = config_field(default_factory=list) # FIXME
    batch_size: int = 256
    eval_batch_size: int = 512
    max_size: Optional[int] = 100
    entry_start: Optional[int] = None
    entry_stop: Optional[int] = None

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = os.getenv('PROJECT_DATA_DIR')

        if len(self.train) == 0:
            self.train.append('perfNano_TTbar_PU200.110X_set0.root')

        if len(self.val) == 0:
            self.val.append('perfNano_TTbar_PU200.110X_set4.root')

    @property
    def train_files(self) -> list[str]:
        return [os.path.join(self.data_dir, each) for each in self.train]

    @property
    def val_files(self) -> list[str]:
        return [os.path.join(self.data_dir, each) for each in self.val]

@dataclass
class TrainingConfig(ConfigBase):
    max_grad_norm: float = 1

@dataclass
class ObjectiveConfig(ConfigBase):
    num_epochs: int = 50

@dataclass
class StudyConfig(ConfigBase):
    n_trials: int = 200
    timeout: Optional[int] = None
    name: str = 'study'


@dataclass
class Config(ConfigBase):
    model: str = 'Transformer'
    data: DataConfig = DataConfig()
    data_transformation: DataTransformationConfig = DataTransformationConfig()
    event_weighting: EventWeightingConfig = EventWeightingConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    training: TrainingConfig = TrainingConfig()
    objective: ObjectiveConfig = ObjectiveConfig()
    study: StudyConfig = StudyConfig()
    cuda: int = config_field(
            default=0,
            help='automatically select an idle gpu')
    num_threads: int = 1
    log_base: Optional[str] = None
    log_name: Optional[str] = None
    mode: str = config_field(default='run', choices=('run', 'sanity-check', 'batch'))

    def __post_init__(self):
        if self.cuda < 0:
            self.cuda = select_idle_gpu(as_idx=True)

        if self.log_base is None:
            self.log_base = os.getenv('PROJECT_LOG_DIR')

        if self.log_name is None:
            now = datetime.now().strftime('%y%m%d-%H%M%S')
            slug = generate_slug(pattern=2)
            self.log_name = f'optuna_{self.mode}_{now}_{slug}'

        if self.mode == 'sanity-check':
            self.data.entry_stop = 2048
            self.study.n_trials = 5
            self.objective.num_epochs = 2

    @property
    def log_dir(self) -> Path:
        return Path(self.log_base) / self.log_name

    @property
    def batch_mode(self) -> bool:
        return self.mode == 'batch'

class Objective(ObjectiveBase):


    def __init__(self,
                 num_epochs: int,
                 model_config_cls: Type[ModelConfigBase],
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 data_xform: DataTransformation,
                 device: torch.device,
                 config: Config
    ) -> None:
        """
        """
        super().__init__(num_epochs=num_epochs)

        self.model_config_cls = model_config_cls

        # attrs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_xform = data_xform
        self.device = device
        self.config = config


    def suggest(self, trial: Trial) -> dict[str, Any]:
        model_config = self.model_config_cls.from_trial(trial)
        loss_name = trial.suggest_categorical(name='loss',
                                         choices=('L1Loss', 'MSELoss', 'HuberLoss'))

        model = model_config.build().to(self.device)
        loss_fn = find_loss_cls(loss_name)(reduction='none').to(self.device)
        optimizer = configure_optimizers(model, **asdict(self.config.optimizer))

        return {
            'model': model,
            'loss_fn': loss_fn,
            'optimizer': optimizer,
        }

    def train(self, suggestion: dict[str, Any]):
        return training.train(
            model=suggestion['model'],
            data_loader=self.train_loader,
            loss_fn=suggestion['loss_fn'],
            optimizer=suggestion['optimizer'],
            device=self.device,
            config=self.config,
            monitor=None,
            lr_scheduler=None,
        )

    def validate(self, suggestion: dict[str, Any]):
        return training.evaluate(
            model=suggestion['model'],
            data_loader=self.val_loader,
            data_xform=self.data_xform,
            loss_fn=suggestion['loss_fn'],
            device=self.device,
            config=self.config,
            phase=training.PhaseEnum.VALIDATION_OPTUNA,
            output_dir=None,
            epoch=None,
        )

    @classmethod
    @property
    def target_name(cls):
        return "reduced_chi2_pt_ratio"

    @classmethod
    @property
    def direction(cls):
        return "minimize"


def run(config: Config):
    print(config)

    log_dir = Path(config.log_dir) # type: ignore
    log_dir.mkdir(parents=True)

    config.to_yaml(log_dir / 'config.yaml')

    ###########################################################################
    # ⭐ sys
    ###########################################################################
    torch.set_num_threads(config.num_threads)

    device = torch.device(f'cuda:{config.cuda}')

    ###########################################################################
    # ⭐ data
    ###########################################################################

    data_xform = Standardization.from_dict(asdict(config.data_transformation))
    event_weighting = config.event_weighting.build()

    train_set = MeteorDataset.from_root(
        path_list=config.data.train_files,
        transformation=data_xform,
        event_weighting=event_weighting,
        entry_start=config.data.entry_start,
        entry_stop=config.data.entry_stop,
    )
    print(f'{len(train_set)=}')

    val_set = MeteorDataset.from_root(
        path_list=config.data.val_files,
        transformation=data_xform,
        event_weighting=event_weighting,
        entry_start=config.data.entry_start,
        entry_stop=config.data.entry_stop,
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
    # ⭐ model, loss function and optimizer
    ###########################################################################
    model_config_cls = find_model_config_cls(config.model)

    ###########################################################################
    # ⭐
    ###########################################################################
    objective = Objective(
        num_epochs=config.objective.num_epochs,
        model_config_cls=model_config_cls,
        train_loader=train_loader,
        val_loader=val_loader,
        data_xform=data_xform,
        device=device,
        config=config,
    )

    run_study(
        objective=objective,
        log_dir=log_dir,
        n_trials=config.study.n_trials,
        timeout=config.study.timeout,
        name=config.study.name,
    )


def main():
    config = Config.from_args()
    run(config)


if __name__ == "__main__":
    main()
