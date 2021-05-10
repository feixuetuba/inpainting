import os
from math import ceil

import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff',
]
def is_img(fpath):
    _, suffix = os.path.splitext(fpath)
    return suffix in IMG_EXTENSIONS

def get_cls(pkg, name):
    import importlib
    pkg = importlib.import_module(f"{pkg}.{name}")
    return getattr(pkg, name)

def show_remain(t):
    value = []
    t = int(t)
    for div in [86400, 3600, 60]:
        value.append(t // div)
        t %= div
    value.append(t)
    desc = []
    for d, v in zip(['D', 'H', 'M', 'S'], value):
        desc.append(f"{v}{d}")
    return ",".join(desc)


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

