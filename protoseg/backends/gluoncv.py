
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data import DataLoader

from protoseg.backends import AbstractBackend
from protoseg.trainer import Trainer


from itertools import chain


class gluoncv(AbstractBackend):

    def __init__(self):
        AbstractBackend.__init__(self)

    def train_epoch(self, trainer):
        dataloader = DataLoader(
            dataset=trainer.dataloader, batch_size=trainer.config['batch_size'])
        for X_batch, y_batch in dataloader:
            print(X_batch.shape, y_batch.shape)
        print('train on gluoncv backend')
