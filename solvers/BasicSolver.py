
import logging
import os
import shutil
import time

import numpy as np
import torch
from dataLoaders import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from utils import get_cls, show_remain
from utils.mask_ref import random_bbox, brush_stroke_mask


class BasicSolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mk_same_mask = cfg['batch_same_mask']
        if self.mk_same_mask:
            self.batch_size = cfg['dataset']['train']['batch_size']
            self.margin_h = cfg['dataset']['train']['vertical_margin']
            self.margin_w = cfg['dataset']['train']['horizontal_margin']
            self.max_delta_h = cfg['dataset']['train']['max_delta_height']
            self.max_delta_w = cfg['dataset']['train']['max_delta_width']
            self.input_size = cfg['input_size']

    def __gen_mask(self, device):

        t, b, l, r = random_bbox(self.input_size, self.input_size,
                                 self.input_size , self.input_size ,
                                 self.margin_h, self.margin_w,
                                 self.max_delta_h, self.max_delta_w)
        regular_mask = torch.zeros((1, 1, self.input_size, self.input_size)).to(device).bool()
        regular_mask[:, t:b, l:r] = True

        irregular_mask = brush_stroke_mask(self.input_size, self.input_size)
        irregular_mask = torch.from_numpy(irregular_mask).to(device).bool()
        mask = torch.bitwise_or(regular_mask, irregular_mask).float()
        mask = mask.repeat((self.batch_size, 1, 1, 1))
        return mask

    def train(self):
        model = get_cls("models", self.cfg['model']['name'])(self.cfg)
        dataloader = get_dataloader(self.cfg, self.cfg['stage'])

        log_dir = os.path.join(self.cfg['checkpoint_dir'], 'log')
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=log_dir)

        which_epoch = self.cfg.get('which_epoch', 0)
        if which_epoch != 0:
            model.load(self.cfg['checkpoint_dir'], self.cfg['nn']['name'], which_epoch)
        step = 0
        logging.info("===== Training... =====")

        epochs = self.cfg['epochs'] + 1
        for epoch in range(which_epoch+1, epochs):
            start = time.time()
            for data in dataloader:
                if self.mk_same_mask:
                    mask = self.__gen_mask(data.device)
                    data = (data, mask)
                model.set_input(*data)
                model.optimize_parameters()
                writer.add_scalars('Loss', model.get_current_error(), step)
                step += 1
            elapse = time.time() - start
            remain = (epochs - epoch) * elapse
            writer.add_image(self.cfg['name'],model.get_current_visual(3), epoch, dataformats='HWC')
            old_lr, lr = model.update_learning_rate()
            logging.info(f"[{epoch}], {old_lr} -> {old_lr}, elapse:{elapse}, remain:{show_remain(remain)}")
            epoch += 1
            if epoch % self.cfg['solver']['save_epoch'] == 0:
                model.save(self.cfg['checkpoint_dir'], self.cfg['nn']['name'], epoch)
        logging.info("===== finished... =====")

    def test(self):
        pass