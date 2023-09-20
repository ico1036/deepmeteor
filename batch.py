#!/usr/bin/env python
from pathlib import Path
import argparse
import shutil
from datetime import datetime
from htcondor import Submit # type: ignore
from htcondor import Schedd # type: ignore
import yaml
from coolname import generate_slug
from deepmeteor import env


def update_config(script_path: Path,
                  input_config_path: Path,
                  sanity_check: bool
) -> tuple[Path, Path]:
    with open(input_config_path) as stream:
        config = yaml.safe_load(stream)

    now = datetime.now().strftime('%y%m%d-%H%M%S')
    slug = generate_slug(pattern=2)

    prefix = script_path.stem
    if sanity_check:
        prefix = f'sanity-check_{prefix}'

    log_base = './'
    log_name = f'{prefix}_{now}_{input_config_path.stem}-{slug}'
    log_dir = Path(log_base) / log_name

    config['log_base'] = log_base
    config['log_name'] = log_name
    if script_path.stem != 'evaluate':
        config['seed'] = -1

    config['batch'] = True
    config['sanity_check'] = sanity_check
    config['cuda'] = 0

    config_path = Path(f'./config/batch/{log_name}.yaml').resolve()
    if not config_path.parent.exists():
        config_path.parent.mkdir(parents=True)
    with open(config_path, 'w') as stream:
        yaml.safe_dump(config, stream)

    return config_path, log_dir


def run(script_path: Path,
        input_config_path: Path,
        sanity_check: bool
):
    """
    """
    if not script_path.exists():
        raise FileNotFoundError(f'{script_path=}')

    if not input_config_path.exists():
        raise FileNotFoundError(f'{input_config_path=}')

    config_path, log_dir = update_config(
        script_path=script_path,
        input_config_path=input_config_path,
        sanity_check=sanity_check
    )

    executable = shutil.which('python')

    arguments = [
        script_path.resolve(),
        config_path.resolve(),
    ]
    arguments = [str(each) for each in arguments]
    arguments = ' '.join(arguments)

    # FIXME use env var
    condor_log_dir = Path('./logs/condor')
    if not condor_log_dir.exists():
        condor_log_dir.mkdir(parents=True)
    condor_log_file = condor_log_dir / log_dir.stem

    submit = {
        'executable': executable,
        'arguments': arguments,
        'universe': 'vanilla',
        'log': condor_log_file.with_suffix('.log'),
        'output': condor_log_file.with_suffix('.out'),
        'error': condor_log_file.with_suffix('.err'),
        'request_cpus': 1,
        'request_gpus': 1,
        'request_memory': '8000M',
        'request_disk': '500M',
        'getenv': True,
        'JobBatchName': f'METEOR.{log_dir.stem}',
        'should_transfer_files': 'YES',
        'when_to_transfer_output': 'ON_EXIT',
        'transfer_output_files': log_dir,
        'transfer_output_remaps': f'"{log_dir} = {env.LOG_DIR}/{log_dir.name}"',
    }
    submit = {key: str(value) for key, value in submit.items()}
    submit = Submit(submit)

    schedd = Schedd()

    result = schedd.submit(submit)
    print(f'{result.num_procs()} jobs submitted with {result.cluster()}')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--script', type=Path,
                        default=(Path.cwd() / 'train.py'),
                        help='Help text')
    parser.add_argument('-c', '--config', type=Path, nargs='+', required=True,
                        help='Help text')
    parser.add_argument('--sanity-check', action='store_true',
                        help='Help text')
    args = parser.parse_args()

    for config in args.config:
        run(
            script_path=args.script,
            input_config_path=config,
            sanity_check=args.sanity_check,
        )


if __name__ == "__main__":
    main()
