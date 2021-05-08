import torch
import torchvision.models as models_baseline # networks with zero padding
import models as models_partial

nn = torch.nn

class PConvModel:
    def __init__(self, cfg):
        m_cfg = cfg.copy()
        m_cfg.update(cfg['model'])
        self.cfg = m_cfg
        self.cfg = m_cfg
        gpu_ids = [int(_) for _ in m_cfg.get('gpu_ids', [])]
        if len(gpu_ids) == 0:
            self.gpu_ids = None
        else:
            self.gpu_ids = [int(_) for _ in gpu_ids]
        self.isTrain = (cfg['stage'] == "train")
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids is not None else torch.device(
            'cpu')  # get device name: CPU or GPU

        nn_cfg = cfg['nn']
        arch = nn_cfg['arch']
        if nn_cfg['pretrained']:
            print("=> using pre-trained model '{}'".format(arch))
            if arch in models_baseline.__dict__:
                model = models_baseline.__dict__[arch](pretrained=True)
            else:
                model = models_partial.__dict__[arch](pretrained=True)
            # model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            if arch in models_baseline.__dict__:
                model = models_baseline.__dict__[arch]()
            else:
                model = models_partial.__dict__[arch]()
        model.to(self.device)
        self.model = model

        if self.isTrain:
            self.model.train()
            self.criterion = nn.CrossEntropyLoss().to(self.device)

            # [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], args.lr,
                                        momentum=m_cfg['momentum'],
                                        weight_decay=m_cfg['weight_decay'])
        else:
            self.model.eval()