#!/usr/bin/env python
import json
from pathlib import Path
import argparse
import pandas as pd

def get_summary(log_dir: Path) -> dict:
    with open(log_dir / 'summary.json') as stream:
        summary = json.load(stream)

    output = {}
    output['name'] = log_dir.name.split('_')[-1]
    for key in ['num_parameters', 'epoch']:
        output[key] = summary[key]

    for key in ['loss', 'reduced_chi2_pt_ratio']:
        output[key] = summary['test'][key]
    return output

def run(log_base_list: list[Path]):
    df = [get_summary(log_dir)
          for log_base in log_base_list
          for log_dir in log_base.glob('run*')
          if (log_dir / 'summary.json').exists()]
    df = pd.DataFrame(df)
    df = df.sort_values(by='reduced_chi2_pt_ratio', ascending=True)
    print(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_bases', type=Path, nargs='+', help='Help text')
    args = parser.parse_args()

    run(args.log_bases)

if __name__ == "__main__":
    main()

