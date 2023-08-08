import typing
from pathlib import Path
import matplotlib.pyplot as plt
from hist.hist import Hist
from deepmeteor.utils import MissingET
from deepmeteor.metrics import Hist1DStat
from deepmeteor.hist import create_hist
from deepmeteor.const import ALGO_LIST
from deepmeteor.const import COLOR_DICT
from deepmeteor.const import ALGO_LABEL_DICT
from deepmeteor.const import get_label


def save_fig(fig: plt.Figure,
             output_path: Path,
             suffix_list: list[str] = ['.png', '.pdf']
):
    fig.tight_layout()
    for suffix in suffix_list:
        fig.savefig(output_path.with_suffix(suffix))


def build_momentum_hist(met_dict: dict[str, MissingET],
                        name: str
) -> dict[str, Hist]:
    hist_dict = {}
    for algo in ALGO_LIST:
        hist_dict[algo] = create_hist(name)
        hist_dict[algo].fill(met_dict[algo][name])
    return hist_dict


def plot_momentum_hist(met_dict: dict[str, MissingET],
                       name: str,
                       output_path: Path | None,
                       close: bool = True,
                       rebin: slice = slice(None),
) -> plt.Figure:
    hist_dict = build_momentum_hist(met_dict, name)

    fig, ax = plt.subplots()
    fig = typing.cast(plt.Figure, fig)
    ax = typing.cast(plt.Axes, ax)

    plot_kwargs = dict(ax=ax, flow='none', lw=2)
    for algo in ALGO_LIST:
        label = ALGO_LABEL_DICT[algo]
        color = COLOR_DICT[algo]
        hist_dict[algo][rebin].plot(label=label, color=color, **plot_kwargs)

    ax.set_xlabel(get_label(name))
    ax.set_ylabel('Events')

    ax.set_ylim(None, 1.2 * ax.get_ylim()[1])

    ax.legend()
    ax.grid()

    if output_path is not None:
        save_fig(fig, output_path)

    if close:
        plt.close(fig)

    return fig


def build_residual_hist(met_dict: dict[str, MissingET],
                        name: str
) -> dict[str, Hist]:
    if name == 'phi':
        residual_func = lambda lhs, rhs: lhs.deltaphi(rhs)
    else:
        residual_func = lambda lhs, rhs: lhs[name] - rhs[name]

    hist_dict = {}
    for algo in ['puppi', 'deep']:
        data = residual_func(met_dict[algo], met_dict['gen'])
        hist_dict[algo] = create_hist(f'residual_{name}')
        hist_dict[algo].fill(data)
    return hist_dict


def plot_residual_hist(met_dict: dict[str, MissingET],
                       name: str,
                       output_path: Path | None,
                       close: bool = True,
                       rebin: slice = slice(None),
) -> plt.Figure:
    hist_dict = build_residual_hist(met_dict, name)

    fig, ax = plt.subplots()
    fig = typing.cast(plt.Figure, fig)
    ax = typing.cast(plt.Axes, ax)

    plot_kwargs = dict(ax=ax, flow='none', lw=2)
    for algo in ['puppi', 'deep']:
        algo_label = ALGO_LABEL_DICT[algo]
        stat = Hist1DStat.from_hist(hist_dict[algo])
        label = f'{algo_label} ({stat})'

        color = COLOR_DICT[algo]

        hist_dict[algo][rebin].plot(label=label, color=color, **plot_kwargs)

    ax.set_xlabel(get_label(f'residual_{name}'))
    ax.set_ylabel('Events')

    ax.set_ylim(None, 1.2 * ax.get_ylim()[1])

    ax.legend()
    ax.grid()

    if output_path is not None:
        save_fig(fig, output_path)

    if close:
        plt.close(fig)

    return fig
