import logging
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.mask_ref import random_bbox, brush_stroke_mask


class FolderDataset(Dataset):
    def __init__(self, cfg, stage):
        super(FolderDataset, self).__init__()
        m_cfg = cfg.copy()
        m_cfg.update(cfg['dataset'])

        self.gen_mask = not m_cfg['batch_same_mask']
        self.input_size = m_cfg['input_size']
        m_cfg.update(m_cfg[stage])
        self.cfg = m_cfg
        self.images = []
        self.__load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = None
        while img is None:
            id = item % len(self.images)
            fpath = self.images[id]
            img = cv2.imread(fpath)
            if img is None:
                self.images.pop(id)
                logging.error(f"Load {fpath} failed")

        if random.uniform(0, 1) < 0.5:
            h, w = img.shape[:2]
            min_dim = min(h, w)
            if min_dim < self.input_size:
                scale = min_dim / self.input_size
                w = self.input_size if w < h else int(w / scale)
                h = self.input_size if w > h else int(h / scale)
                img = cv2.resize(img, (w, h))
            delta_w = w - self.input_size
            delta_h = h - self.input_size
            l = t = 0
            if delta_w > 0:
                l = random.randint(0, delta_w)
            if delta_h > 0:
                t = random.randint(0, delta_h)
            img = img[t:, l:]
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = np.transpose(img, (2,0,1)).astype(float)
        img = img.astype(float) / 127.5 - 1
        if self.gen_mask:
            t,b, l, r = random_bbox(self.input_size, self.input_size,
                                self.input_size, self.input_size,
                                self.cfg['vertical_margin'], self.cfg['horizontal_margin'],
                                self.cfg['max_delta_height'], self.cfg['max_delta_width'])
            regular_mask = np.zeros((1, self.input_size, self.input_size), np.uint8)
            regular_mask[:, t:b, l:r] = 1
            irregular_mask = brush_stroke_mask(self.input_size, self.input_size)
            mask = cv2.bitwise_or(regular_mask, irregular_mask).astype(np.float32)
            return img, mask
        else:
            return img

    def __load_images(self):
        file_root = self.cfg['dataroot']
        with open(self.cfg['filelist'], "r") as fd:
            for line in fd.readlines():
                if line.startswith("#"):
                    continue
                full_path = os.path.join(file_root, line.strip())
                if not os.path.isfile(full_path):
                    continue
                self.images.append(full_path)
