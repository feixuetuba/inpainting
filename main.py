import logging
import os.path
from argparse import ArgumentParser

import yaml

from utils import get_cls

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("cfg", help="configuration file")
    parser.add_argument("--stage", default="test")
    parser.add_argument("--which_epoch", type=int, default=-1)
    parser.add_argument("--checkpoints", default=None, help="checkpoints dir")
    parser.add_argument("--imgs", default=None, help="img dir for test")
    parser.add_argument("--dest", default="test_results", help="dest dir to save results")
    opts = parser.parse_args()

    os.environ['TORCH_HOME'] = r'D:\model_zoo'

    with open(opts.cfg, "r") as fd:
        config = yaml.load(fd, yaml.FullLoader)
    config['stage'] = opts.stage
    config['which_epoch'] = opts.which_epoch
    if opts.checkpoints is None:
        config['checkpoint_dir'] = os.path.join("checkpoints", config['name'])
    else:
        config['checkpoint_dir'] = opts.checkpoints

    if opts.stage == "test":
        ds_cfg =config['dataset']
        if 'test' not in ds_cfg:
            ds_cfg['test'] = {}
        ds_cfg['test']['dataroot'] = opts.imgs
        ds_cfg['test']['dest'] = opts.dest
    else:
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
    solver = get_cls('solvers', config['solver']['name'])(config)
    getattr(solver, opts.stage)()
