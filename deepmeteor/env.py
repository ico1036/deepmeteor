import os
from typing import cast
from pathlib import Path

DATA_DIR = cast(str, os.getenv('PROJECT_DATA_DIR'))
DATA_DIR_FALLBACK = cast(str, os.getenv('PROJECT_DATA_DIR_FALLBACK'))
LOG_DIR = cast(str, os.getenv('PROJECT_LOG_DIR'))

def find_from_data_dir(name: str) -> Path:
    data_dir = Path(DATA_DIR)

    file = data_dir / name
    if not file.exists():
        # TODO warning
        file = Path(DATA_DIR_FALLBACK) / name

    if not file.exists():
        raise FileNotFoundError

    return file
