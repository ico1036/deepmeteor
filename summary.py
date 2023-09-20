#!/usr/bin/env python
import json
from pathlib import Path
import argparse
import yaml
import pandas as pd



METRIC = 'pT 𝛘2/ν ratio'

def _get_summary(log_dir: Path) -> dict:
    with open(log_dir / 'summary.json') as stream:
        summary = json.load(stream)

    with open(log_dir / 'config.yaml') as stream:
        config = yaml.safe_load(stream)

    try:
        *_, date, name = log_dir.name.split('_')
        date, _ = date.split('-')
    except Exception as error:
        print(log_dir)
        raise error

    output = {}
    # output['date'] = date
    output['Name'] = name
    output['Model'] = config['model']['name']
    output['Params'] = summary['num_parameters']
    output['Epoch'] = f'{summary["epoch"]}/{config["training"]["num_epochs"]}'
    output[METRIC] = summary['test']['reduced_chi2_pt_ratio']
    output[METRIC + ' @ val'] = summary['validation']['reduced_chi2_pt_ratio']
    #output['LR'] = config['optimizer']['learning_rate']
    output['loss_100_150'] = summary['test']['loss_100_150']

    return output

def get_summary(log_dir: Path):
    try:
        output = _get_summary(log_dir)
    except Exception as error:
        print(log_dir)
        raise error
    return output

def run(log_base_list: list[Path], nrows: int):
    df = [get_summary(log_dir)
          for log_base in log_base_list
          for log_dir in log_base.glob('*')
          if (log_dir / 'summary.json').exists() and 'sanity-check' not in log_dir.stem]
    df = pd.DataFrame(df)
    df = df.sort_values(by=METRIC, ascending=True)
    # df = df.sort_values(by=METRIC, ascending=True)

    print(f'showing {nrows} results out of {len(df)}...', end='\n\n')

    print(df.head(n=nrows).to_markdown(index=False))
    # print(df.head(n=nrows).to_string(index=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', dest='log_base_list', type=Path,
                        nargs='+', default=[Path.cwd() / 'logs'],
                        help='Help text')
    parser.add_argument('-n', '--nrows', default=10, type=int, help='Help text')
    args = parser.parse_args()

    run(**vars(args))

if __name__ == "__main__":
    main()

