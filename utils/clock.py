import logging
import time

import numpy as np


class Clock:
    def __init__(self):
        self.tick_tocks = {}

    def tick(self, name="default"):
        if name not in self.tick_tocks:
            self.tick_tocks[name] = {'latest':0, 'elapse':[]}
        self.tick_tocks[name]['latest'] = time.time()

    def tock(self, name="default"):
        if name in self.tick_tocks:
            elapse = time.time() - self.tick_tocks[name]['latest']
            self.tick_tocks[name]['elapse'] = elapse

    def clean(self, name):
        self.tick_tocks[name] = {'latest': 0, 'elapse': []}

    def summery(self):
        ret = []
        for name, info in self.tick_tocks.items():
            ret.append((name, np.mean(self.tick_tocks['elapse'])))
        return ret