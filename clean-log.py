#!/usr/bin/env python
import shutil
from pathlib import Path

def is_invalid_log_dir(path: Path) -> bool:
    if path.is_file() or path.name == 'condor':
        return False

    summary_path = path / 'summary.json'
    return not summary_path.exists()



log_base_list = [Path('./logs/')]

log_dir_list = [log_dir
                for log_base in log_base_list
                for log_dir in log_base.glob('*')
                if is_invalid_log_dir(log_dir)]

for log_base in log_base_list:
    for log_dir in log_base.glob('train_*'):
        if is_invalid_log_dir(log_dir):
            print(f'rmtree {log_dir}')
            shutil.rmtree(str(log_dir))
