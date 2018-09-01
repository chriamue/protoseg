from __future__ import absolute_import
import os
import numpy as np
import cv2
import mxnet
from mxnet import nd
from mxnet import gluon, autograd
from mxnet.gluon.data import DataLoader
import gluoncv
from gluoncv.model_zoo.segbase import SoftmaxCrossEntropyLossWithAux
from gluoncv.utils.parallel import DataParallelModel, DataParallelCriterion

from protoseg.backends import AbstractBackend
from protoseg.trainer import Trainer



class gluoncv_backend(AbstractBackend):
    ctx = mxnet.gpu()
    ctx_list = [ctx]

    def __init__(self):
        AbstractBackend.__init__(self)

    def load_model(self, config, modelfile):
        model = gluoncv.model_zoo.get_fcn(
            dataset='pascal_voc', backbone=config['backbone'], pretrained=config['pretrained'], ctx=self.ctx_list)
        model.hybridize()
        if os.path.isfile(modelfile):
            print('loaded model from:', modelfile)
            model.load_parameters(modelfile, ctx=self.ctx)
        return model

    def save_model(self, model):
        model.model.module.save_parameters(model.modelfile)
        print('saved model to:', model.modelfile)

    def init_trainer(self, trainer):
        trainer.loss_function = SoftmaxCrossEntropyLossWithAux(aux=True)
        trainer.lr_scheduler = gluoncv.utils.LRScheduler(mode='poly', baselr=trainer.config['learn_rate'], niters=len(trainer.dataloader),
                                                         nepochs=50)
        trainer.model.model = DataParallelModel(trainer.model.model, self.ctx_list)
        trainer.loss_function = DataParallelCriterion(trainer.loss_function, self.ctx_list)
        kv = mxnet.kv.create('local')
        trainer.optimizer = gluon.Trainer(trainer.model.model.module.collect_params(), 'sgd',
                                             {'lr_scheduler': trainer.lr_scheduler,
                                              'wd': 0.0001,
                                              'momentum': 0.9,
                                              'multi_precision': True},
                                              kvstore=kv)

    def dataloader_format(self, img, mask):
        
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = np.rollaxis(img, axis=2, start=0)
        return mxnet.nd.array(img), mxnet.nd.array(mask)

    def train_epoch(self, trainer):
        batch_size = trainer.config['batch_size']
        dataloader = DataLoader(
            dataset=trainer.dataloader, batch_size=batch_size, last_batch='rollover', num_workers=batch_size)
        for i, (X_batch, y_batch) in enumerate(dataloader):
            trainer.lr_scheduler.update(i, trainer.epoch)
            with autograd.record(True):
                outputs = trainer.model.model(X_batch)
                losses = trainer.loss_function(outputs, y_batch)
                mxnet.nd.waitall()
                autograd.backward(losses)
            trainer.optimizer.step(batch_size)
            for loss in losses:
                trainer.loss += loss.asnumpy()[0]
        print('train on gluoncv backend')
