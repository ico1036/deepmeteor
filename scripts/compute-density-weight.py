#!/usr/bin/env python
from pathlib import Path
import os
import argparse
import uproot.writing
from deepmeteor.data.eventweighting import DensityWeightHist


def run(input_dir: Path, output_dir: Path):
    path_list = [input_dir / f'perfNano_TTbar_PU200.110X_set{index}.root'
                 for index in range(4)]

    output_path = output_dir / DensityWeightHist.__name__

    event_weighting = DensityWeightHist.from_root(path_list)

    event_weighting.to_npz(output_path.with_suffix('.npz'))

    file = uproot.writing.create(output_path.with_suffix('.root'))
    file['weight'] = event_weighting.to_hist()
    file.close()

    return event_weighting

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', default=None, type=Path, help='Help text')
    parser.add_argument('-o', '--output-dir', default=None, type=Path, help='Help text')
    args = parser.parse_args()

    args.input_dir = args.input_dir or Path(os.getenv('PROJECT_DATA_DIR'))
    args.output_dir = args.output_dir or args.input_dir

    print(f'input_dir: {args.input_dir}')
    print(f'output_dir: {args.output_dir}')

    run(input_dir=args.input_dir,
        output_dir=args.output_dir)


if __name__ == "__main__":
    main()
