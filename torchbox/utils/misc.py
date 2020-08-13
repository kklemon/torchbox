from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch


def get_default_device(default: Optional[Union[str, torch.device]] = None):
    if default:
        device = default
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def setup_run(run_name=None, results_dir='results', create_dirs=('checkpoints', 'samples'), datefmt='%Y-%m-%d_%H-%M-%S'):
    if not run_name:
        run_name = datetime.now().strftime(datefmt)

    results_dir = Path(results_dir)

    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True)

    for d in create_dirs:
        (run_dir / d).mkdir()

    return run_dir
