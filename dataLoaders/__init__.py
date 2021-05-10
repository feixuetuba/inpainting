from torch.utils.data import DataLoader

from utils import get_cls

def get_dataloader(cfg, stage='train'):
    ds_cfg = cfg['dataset'][stage]
    name = cfg['dataset']['name']
    dataset = get_cls("dataLoaders", name)(cfg, stage)
    return DataLoader(dataset, ds_cfg['batch_size'], ds_cfg['shuffle'],
                      num_workers=ds_cfg['n_workers'],drop_last=ds_cfg.get('drop_last', False))
