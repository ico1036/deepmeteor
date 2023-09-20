#!/usr/bin/env python
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as mh
from deepmeteor.utils import Errorbar
from deepmeteor.env import find_from_data_dir
from deepmeteor.env import LOG_DIR
mpl.use('agg')
mh.style.use(mh.styles.CMS)


def run(log_dir: Path, include_sanity_check: bool):
    errorbar_dict: dict[str, Errorbar] = {}

    baseline_path_list = [
        find_from_data_dir('gen-vs-rec/DeepMET.npz'),
        find_from_data_dir('gen-vs-rec/PUPPI.npz'),
    ]

    for path in baseline_path_list:
        errorbar_dict[path.stem] = Errorbar.from_npz(path)

    for each in log_dir.glob('*'):
        if not include_sanity_check:
            if each.name.startswith('sanity-check'):
                continue
        path = each / 'eval' / 'test' / 'hist' / 'gen-vs-rec.npz'
        if not path.exists():
            # warning
            continue
        *_, _, name = each.stem.split('_')
        errorbar_dict[name] = Errorbar.from_npz(path)

    fig, ax = plt.subplots()
    for label, errorbar in errorbar_dict.items():
        errorbar.plot(ax=ax, label=label)
    ax.legend()

    xlow, xup = 0, 500
    ax.plot([xlow, xup], [xlow, xup], lw=2, alpha=0.8, ls='--', color='gray')
    ax.set_xlim(xlow, xup)
    ax.set_ylim(xlow, xup)

    ax.set_xlabel(r'Generated $p_{T}^{miss}\ [GeV]$')
    ax.set_ylabel(r'Reconstructed $p_{T}^{miss}\ [GeV]$')

    output_path = log_dir / 'gen-vs-rec'
    for suffix in ['.png', '.pdf']:
        fig.savefig(output_path.with_suffix(suffix))


def main():
    log_dir = Path(LOG_DIR)
    include_sanity_check = False
    run(log_dir, include_sanity_check)

if __name__ == "__main__":
    main()
