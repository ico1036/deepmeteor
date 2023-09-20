#!/usr/bin/env python3
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils.clip_grad import clip_grad_norm_
import tqdm
import matplotlib as mpl
import mplhep as hep
import uproot
import uproot.writing
from coolname import generate_slug
from torchhep.optim import configure_optimizers
from torchhep.utils.checkpoint import Checkpoint
from torchhep.utils.cuda import select_idle_gpu
from torchhep.utils.reproducibility import sample_seed
from torchhep.utils.reproducibility import set_seed
from torchhep.utils.earlystopping import EarlyStopping
from hierconfig.config import config_field, ConfigBase
from deepmeteor.data.dataset import MeteorDataset
from deepmeteor.data.transformations import DataTransformation
from deepmeteor.data.transformations import Standardization
from deepmeteor.data.eventweighting import EventWeightingConfig
from deepmeteor.models.base import ModelConfigBase
from deepmeteor.models.base import ModelBase
from deepmeteor.models.utils import init_weights
from deepmeteor.losses.utils import find_loss_cls
from deepmeteor.utils import Errorbar, MissingET
from deepmeteor.result import TrainingResult
from deepmeteor.result import EvaluationResult
from deepmeteor.result import EpochResult
from deepmeteor.result import Summary
from deepmeteor.learningcurve import Monitor
from deepmeteor.hist import create_hist
from deepmeteor.utils import compute_residual
from deepmeteor.plot import plot_gen_vs_rec, plot_momentum_hist
from deepmeteor.plot import plot_residual_hist
from deepmeteor import metrics
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
        self.data_dir = self.data_dir or env.DATA_DIR

    @property
    def train_files(self) -> list[str]:
        return [os.path.join(self.data_dir, each) for each in self.train]

    @property
    def val_files(self) -> list[str]:
        return [os.path.join(self.data_dir, each) for each in self.val]

    @property
    def test_files(self) -> list[str]:
        return [os.path.join(self.data_dir, each) for each in self.test]


@dataclass
class DataTransformationConfig(ConfigBase):
    puppi_cands_cont_std: list[float] = config_field(default_factory=lambda: [10.68, 10.68, 1.42, 1.00])
    gen_met_std: list[float] = config_field(default_factory=lambda: [64.98, 64.98])


@dataclass
class OptimizerConfig(ConfigBase):
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    fused: bool = True


@dataclass
class TrainingConfig(ConfigBase):
    loss: str = config_field(default='HuberLoss',
                             choices=('L1Loss',' MSELoss', 'HuberLoss'))
    num_epochs: int = 10
    max_grad_norm: float = 1
    num_epochs: int = 10
    early_stopping_patience: int = 20
    lr_cosine_annealing: bool = False
    t_0: int = 10
    t_mult: int = 2


@dataclass
class RunConfigBase(ConfigBase):
    model: ModelConfigBase
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
    compile: bool = False

    @property
    def mode(self):
        mode = 'train'
        if self.sanity_check:
            mode = f'sanity-check_{mode}'
        return mode

    def __post_init__(self):
        if self.cuda < 0:
            self.cuda = select_idle_gpu(as_idx=True)

        self.log_base = self.log_base or env.LOG_DIR

        if self.log_name is None:
            now = datetime.now().strftime('%y%m%d-%H%M%S')
            slug = generate_slug(pattern=2)
            self.log_name = f'{self.mode}_{now}_{slug}'

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
    def log_dir(self) -> Path:
        return Path(self.log_base) / self.log_name


class PhaseEnum(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3
    VALIDATION_OPTUNA = 4


def train(model: ModelBase,
          data_loader: DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          config: RunConfigBase,
          monitor: Monitor | None,
          lr_scheduler: CosineAnnealingWarmRestarts | None = None,
          epoch: int = -10000,
) -> None:
    model.train()

    num_batches = len(data_loader)
    progress_bar = tqdm.tqdm(data_loader, disable=config.batch)
    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model.run(batch)
        loss = loss_fn(input=output, target=batch.target)
        # FIXME
        loss = (batch.weight * loss.mean(dim=1)).mean()
        loss.backward()
        if config.training.max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()

        # FIXME
        if monitor is not None:
            monitor.training.append(TrainingResult(
                loss=loss.item()
            ))

        if lr_scheduler is not None:
            # epoch starts with 1
            lr_scheduler.step((epoch - 1) + batch_idx / num_batches)

        progress_bar.set_description(f'🏋️ [TRAINING] {loss=:.6f}')


@torch.inference_mode()
def evaluate(model: ModelBase,
             data_loader: DataLoader,
             data_xform: DataTransformation,
             loss_fn: nn.Module,
             device: torch.device,
             config: RunConfigBase,
             phase: PhaseEnum,
             output_dir: Path | None,
             epoch: int | None,
) -> EvaluationResult:
    ###########################################################################
    # setup
    ###########################################################################
    model.eval()

    ###########################################################################
    #
    ###########################################################################
    loss_sum = 0
    gen_met_edge_list = [0, 30, 60, 100, 150, float('inf')]
    gen_met_range_list = list(zip(gen_met_edge_list[:-1],
                                  gen_met_edge_list[1:]))
    loss_sum_dict = {f'{low}_{up}': 0 for low, up in gen_met_range_list}
    example_count_dict = {f'{low}_{up}': 0 for low, up in gen_met_range_list}

    if phase is PhaseEnum.TEST:
        output_path = output_dir / 'test.root'
        output_file = uproot.writing.create(output_path)
        branch_types = {f'{algo}_{var}': 'float32'
                        for algo in ['gen', 'puppi', 'meteor']
                        for var in ['pt', 'phi']}
        output_file.mktree('tree', branch_types)
    elif phase is PhaseEnum.VALIDATION:
        # only for histograms
        output_path = output_dir / 'validation.root'
        output_file = uproot.writing.update(output_path)
        output_file.mkdir(f'epoch_{epoch:0>5d}')
    elif phase is PhaseEnum.VALIDATION_OPTUNA:
        ...
    else:
        raise RuntimeError

    # momentum components
    component_list = ['px', 'py', 'pt', 'phi']
    # momentum histograms
    mom_hists = {f'{algo}_{comp}': create_hist(comp)
                 for algo in ['gen', 'puppi', 'meteor']
                 for comp in component_list}
    # residual histograms
    res_hists = {f'{algo}_{comp}': create_hist(comp)
                 for algo in ['puppi', 'meteor']
                 for comp in component_list}

    ###########################################################################
    # loop
    ###########################################################################
    for batch in tqdm.tqdm(data_loader, desc=f'👀 [{phase.name}]',
                           disable=config.batch):
        batch = batch.to(device)
        output = model.run(batch)
        loss = loss_fn(input=output, target=batch.target)
        raw_loss = loss.mean(dim=1)
        loss = batch.weight * raw_loss

        #######################################################################
        #
        #######################################################################
        met_dict = {
            'gen': batch.target,
            'puppi': batch.puppi_met,
            'meteor': output,
        }

        met_dict = {key: data_xform.inverse_transform_gen_met(value.cpu())
                    for key, value in met_dict.items()}

        met_dict = {key: MissingET.from_tensor(value)
                    for key, value in met_dict.items()}

        #######################################################################
        # accumulation
        #######################################################################
        loss_sum += loss.sum().item()
        for pt_min, pt_max in gen_met_range_list:
            mask = (batch.gen_met_pt > pt_min) & (batch.gen_met_pt < pt_max)
            key = f'{pt_min}_{pt_max}'
            loss_sum_dict[key] += loss[mask].sum().item()
            example_count_dict[key] += mask.count_nonzero().item()

        for comp in component_list:
            for algo in ['gen', 'puppi', 'meteor']:
                mom_hists[f'{algo}_{comp}'].fill(met_dict[algo][comp])

            for algo in ['puppi', 'meteor']:
                residual = compute_residual(met_dict[algo], met_dict['gen'],
                                            comp)
                res_hists[f'{algo}_{comp}'].fill(residual)

        if phase is PhaseEnum.TEST:
            output_chunk = {}
            for comp in ['pt', 'phi']:
                for algo, met in met_dict.items():
                    output_chunk[f'{algo}_{comp}'] = met[comp]
            output_file['tree'].extend(output_chunk)

    ###########################################################################
    #
    ###########################################################################

    if phase is PhaseEnum.VALIDATION:
        for comp in ['pt', 'phi']:
            output_file[f'epoch_{epoch:0>5d}/{comp}'] = mom_hists[f'meteor_{comp}']
        output_file.close()
    elif phase is PhaseEnum.TEST:
        output_file.close()
    elif phase is PhaseEnum.VALIDATION_OPTUNA:
        ...
    else:
        raise RuntimeError


    loss_dict = {f'loss_{key}': loss_sum / example_count_dict[key]
                 for key, loss_sum in loss_sum_dict.items()}

    result_kwargs = {
        'loss': loss_sum / len(data_loader.dataset)
    }
    result_kwargs |= loss_dict

    # reduced chi2
    for comp in component_list:
        deep_reduced_chi2 = metrics.compute_reduced_chi2(
            mom_hists[f'meteor_{comp}'],
            mom_hists[f'gen_{comp}']
        )
        puppi_reduced_chi2 = metrics.compute_reduced_chi2(
            mom_hists[f'puppi_{comp}'],
            mom_hists[f'gen_{comp}']
        )

        result_kwargs |= {
            f'reduced_chi2_{comp}': deep_reduced_chi2,
            f'reduced_chi2_{comp}_ratio': deep_reduced_chi2 / puppi_reduced_chi2
        }

    # mean and std dev of residuals
    stats = {key: metrics.Hist1DStat.from_hist(value)
             for key, value in res_hists.items()}
    for comp in component_list:
        result_kwargs[f'residual_{comp}_mean'] = stats[f'meteor_{comp}'].mean
        result_kwargs[f'residual_{comp}_std'] = stats[f'meteor_{comp}'].std
        # relative
        std_ratio = stats[f'meteor_{comp}'].std / stats[f'puppi_{comp}'].std
        result_kwargs[f'residual_{comp}_std_ratio'] = std_ratio

    result = EvaluationResult(**result_kwargs)
    return result


def make_gen_vs_rec_errorbar(rec_met: MissingET,
                             gen_met: MissingET,
                             pt_range: np.ndarray | None = None,
):
    pt_range = pt_range or np.linspace(0, 500, 21)

    gen_mask_list = [(gen_met.pt >= low) & (gen_met.pt < up)
                     for low, up in zip(pt_range[:-1], pt_range[1:])]

    mean = np.array([rec_met.pt[mask].mean() for mask in gen_mask_list])
    std = np.array([rec_met.pt[mask].std() for mask in gen_mask_list])

    centre = (pt_range[:-1] + pt_range[1:]) / 2
    half_width = (pt_range[1:] - pt_range[:-1]) / 2

    return Errorbar(x=centre, y=mean, xerr=half_width, yerr=std)


def save_histogram_plots(input_path: Path,
                         output_dir: Path,
):
    tree = uproot.open(f'{input_path}:tree')
    met_dict = {algo: MissingET.from_tree(tree, algo)
                for algo in ['gen', 'puppi', 'meteor']}

    for name in ['px', 'py', 'pt', 'phi']:
        plot_momentum_hist(met_dict, name, output_path=(output_dir / name))
        plot_residual_hist(met_dict, name,
                           output_path=(output_dir / f'residual_{name}'))

    # gen vs rec
    meteor_errorbar = make_gen_vs_rec_errorbar(rec_met=met_dict['meteor'],
                                               gen_met=met_dict['gen'])
    gen_vs_rec_path = output_dir / 'gen-vs-rec'
    meteor_errorbar.to_npz(gen_vs_rec_path.with_suffix('.npz'))
    plot_gen_vs_rec(meteor_errorbar, output_path=gen_vs_rec_path)


def run(config: RunConfigBase) -> None:
    mpl.use('agg')
    hep.style.use(hep.styles.CMS)

    log_dir = config.log_dir
    log_dir.mkdir(parents=True)
    config.to_yaml(log_dir / 'config.yaml')

    print(str(config))

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
    model = config.model.build().to(device)
    model.apply(init_weights)
    if config.compile:
        model = torch.compile(model)
    print(model)
    print(f'# of parameters = {model.num_parameters}')

    # optimizer
    optimizer = configure_optimizers(model, **asdict(config.optimizer))
    print(optimizer)

    if config.training.lr_cosine_annealing:
        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=config.training.t_0,
            T_mult=config.training.t_mult
        )
    else:
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
        cut=config.data.train_cut,
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

    early_stopping = EarlyStopping(
        direction='minimize',
        patience=config.training.early_stopping_patience
    )

    with uproot.writing.create(learning_curve_dir / 'validation.root'):
        ...

    for epoch in range(0, 1 + config.training.num_epochs):
        print(f'\n🔥 [{datetime.now()}] Epoch {epoch}')

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
        monitor.to_json(learning_curve_dir)
        if epoch >= 1:
            monitor.draw_all(output_dir=learning_curve_dir)

        print(str(val_result))

        checkpoint.step(loss=val_result.loss, epoch=epoch)
        if val_result.loss < summary.validation.loss:
            summary.step = monitor.last_step
            summary.epoch = epoch
            summary.validation = val_result

        if early_stopping.step(metric=val_result.loss):
            print('[EarlyStopping] stop training')
            break

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
        cut=config.data.eval_cut,
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
