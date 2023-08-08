#!/usr/bin/env python3
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
from enum import Enum
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
from hist import Hist
from torchhep.optim import configure_optimizers
from torchhep.utils.checkpoint import Checkpoint
from torchhep.utils.cuda import select_idle_gpu
from torchhep.utils.reproducibility import sample_seed
from torchhep.utils.reproducibility import set_seed
from hierconfig.config import config_field, ConfigBase
from deepmeteor.data.dataset import MeteorDataset
from deepmeteor.data.transformations import DataTransformation
from deepmeteor.data.transformations import Standardization
from deepmeteor.models.base import ModelConfigBase
from deepmeteor.models.base import ModelBase
from deepmeteor.models.utils import init_weights
from deepmeteor.losses.utils import find_loss_cls
from deepmeteor.utils import MissingET
from deepmeteor.result import TrainingResult
from deepmeteor.result import EvaluationResult
from deepmeteor.result import EpochResult
from deepmeteor.result import Summary
from deepmeteor.learningcurve import Monitor
from deepmeteor.hist import create_hist
from deepmeteor.utils import compute_residual
from deepmeteor.plot import plot_momentum_hist
from deepmeteor.plot import plot_residual_hist
from deepmeteor import metrics


@dataclass
class DataConfig(ConfigBase):
    data_dir: Optional[str] = None
    train: list[str] = config_field(default_factory=list) # FIXME
    val: list[str] = config_field(default_factory=list) # FIXME
    test: list[str] = config_field(default_factory=list) # FIXME
    batch_size: int = 256
    eval_batch_size: int = 512
    max_size: Optional[int] = 100
    entry_start: Optional[int] = None
    entry_stop: Optional[int] = None

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = os.getenv('PROJECT_DATA_DIR')

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
    lr_cosine_annealing: bool = False
    t_0: int = 10
    t_mult: int = 2


@dataclass
class RunConfigBase(ConfigBase):
    model: ModelConfigBase
    data: DataConfig = DataConfig()
    data_transformation: DataTransformationConfig = DataTransformationConfig()
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
    mode: str = config_field(default='run', choices=('run', 'sanity-check'))

    def __post_init__(self):
        if self.cuda < 0:
            self.cuda = select_idle_gpu(as_idx=True)

        if self.log_base is None:
            self.log_base = os.getenv('PROJECT_LOG_DIR')

        if self.log_name is None:
            now = datetime.now().strftime('%y%m%d-%H%M%S')
            slug = generate_slug(pattern=2)
            self.log_name = f'{self.mode}_{now}_{slug}'

        if self.seed < 0:
            self.seed = sample_seed()

        if self.mode == 'sanity-check':
            self.training.num_epochs = 2

    @property
    def log_dir(self) -> Path:
        return Path(self.log_base) / self.log_name


class PhaseEnum(Enum):
    TRAINING = 1
    VALIDATION = 2
    TEST = 3


def train(model: ModelBase,
          data_loader: DataLoader,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          config: RunConfigBase,
          monitor: Monitor,
          lr_scheduler: CosineAnnealingWarmRestarts | None = None,
          epoch: int = -10000,
) -> None:
    model.train()

    num_batches = len(data_loader)
    progress_bar = tqdm.tqdm(data_loader)
    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model.run(batch)
        loss = loss_fn(input=output, target=batch.target)
        loss.backward()
        clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()

        # FIXME
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
             phase: PhaseEnum,
             output_dir: Path,
             config: RunConfigBase,
             epoch: int,
) -> EvaluationResult:
    ###########################################################################
    # setup
    ###########################################################################
    model.eval()

    ###########################################################################
    #
    ###########################################################################
    loss_sum = 0

    if phase is PhaseEnum.TEST:
        output_path = output_dir / 'test.root'
        output_file = uproot.writing.create(output_path)
        branch_types = {f'{algo}_{var}': 'float32'
                        for algo in ['gen', 'puppi', 'deep']
                        for var in ['pt', 'phi']}
        output_file.mktree('tree', branch_types)
    elif phase is PhaseEnum.VALIDATION:
        # only for histograms
        output_path = output_dir / 'validation.root'
        output_file = uproot.writing.update(output_path)
        output_file.mkdir(f'epoch_{epoch:0>5d}')
    else:
        raise RuntimeError

    # momentum components
    component_list = ['px', 'py', 'pt', 'phi']
    # momentum histograms
    mom_hists = {f'{algo}_{comp}': create_hist(comp)
                 for algo in ['gen', 'puppi', 'deep']
                 for comp in component_list}
    # residual histograms
    res_hists = {f'{algo}_{comp}': create_hist(comp)
                 for algo in ['puppi', 'deep']
                 for comp in component_list}

    ###########################################################################
    # loop
    ###########################################################################
    for batch in tqdm.tqdm(data_loader, desc=f'👀 [{phase.name}]'):
        batch = batch.to(device)
        output = model.run(batch)
        loss = loss_fn(input=output, target=batch.target)

        #######################################################################
        #
        #######################################################################
        met_dict = {
            'gen': batch.target,
            'puppi': batch.puppi_met,
            'deep': output,
        }

        met_dict = {key: data_xform.inverse_transform_gen_met(value.cpu())
                    for key, value in met_dict.items()}

        met_dict = {key: MissingET.from_tensor(value)
                    for key, value in met_dict.items()}

        #######################################################################
        # accumulation
        #######################################################################
        loss_sum += len(batch) * loss.item()

        for comp in component_list:
            for algo in ['gen', 'puppi', 'deep']:
                mom_hists[f'{algo}_{comp}'].fill(met_dict[algo][comp])

            for algo in ['puppi', 'deep']:
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
            output_file[f'epoch_{epoch:0>5d}/{comp}'] = mom_hists[f'deep_{comp}']
        output_file.close()
    elif phase is PhaseEnum.TEST:
        output_file.close()
    else:
        raise RuntimeError

    result_kwargs = {
        'loss': loss_sum / len(data_loader.dataset)
    }

    # reduced chi2
    for comp in component_list:
        deep_reduced_chi2 = metrics.compute_reduced_chi2(
            mom_hists[f'deep_{comp}'],
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
        result_kwargs[f'residual_{comp}_mean'] = stats[f'deep_{comp}'].mean
        result_kwargs[f'residual_{comp}_std'] = stats[f'deep_{comp}'].std
        # relative
        result_kwargs[f'residual_{comp}_std_ratio'] = stats[f'deep_{comp}'].std / stats[f'puppi_{comp}'].std

    result = EvaluationResult(**result_kwargs)
    return result


def save_histogram_plots(input_path: Path,
                         output_dir: Path,
):
    tree = uproot.open(f'{input_path}:tree')
    met_dict = {algo: MissingET.from_tree(tree, algo)
                for algo in ['gen', 'puppi', 'deep']}

    for name in ['px', 'py', 'pt', 'phi']:
        plot_momentum_hist(met_dict, name, output_path=(output_dir / name))
        plot_residual_hist(met_dict, name,
                           output_path=(output_dir / f'residual_{name}'))

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
    # model = torch.compile(model)
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
    loss_fn = find_loss_cls(config.training.loss)().to(device)

    ###########################################################################
    # ⭐ dataset
    ###########################################################################
    # TODO
    data_xform = Standardization.from_dict(asdict(config.data_transformation))

    train_set = MeteorDataset.from_root(
        path_list=config.data.train_files,
        transformation=data_xform,
        entry_start=config.data.entry_start,
        entry_stop=config.data.entry_stop,
    )
    print(f'{len(train_set)=}')

    val_set = MeteorDataset.from_root(
        path_list=config.data.val_files,
        transformation=data_xform,
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

    uproot.writing.create(learning_curve_dir / 'validation.root')

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
        entry_start=config.data.entry_start,
        entry_stop=config.data.entry_stop,
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
