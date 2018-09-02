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
from gluoncv.data import batchify

from protoseg.backends import AbstractBackend
from protoseg.trainer import Trainer

from mxboard import SummaryWriter


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
        trainer.model.model = DataParallelModel(
            trainer.model.model, self.ctx_list)
        trainer.loss_function = DataParallelCriterion(
            trainer.loss_function, self.ctx_list)
        kv = mxnet.kv.create('local')
        trainer.optimizer = gluon.Trainer(trainer.model.model.module.collect_params(), 'sgd',
                                          {'lr_scheduler': trainer.lr_scheduler,
                                              'wd': 0.0001,
                                              'momentum': 0.9,
                                              'multi_precision': True},
                                          kvstore=kv)

    def dataloader_format(self, img, mask=None):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.rollaxis(img, axis=2, start=0)
        if mask is None:
            return mxnet.nd.array(img)

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return mxnet.nd.array(img), mxnet.nd.array(mask)

    def train_epoch(self, trainer):
        batch_size = trainer.config['batch_size']
        summarysteps = trainer.config['summarysteps']

        dataloader = DataLoader(
            dataset=trainer.dataloader, batch_size=batch_size, last_batch='rollover', num_workers=batch_size)

        for i, (X_batch, y_batch) in enumerate(dataloader):
            trainer.global_step += 1
            trainer.lr_scheduler.update(i, trainer.epoch)
            with autograd.record(True):
                outputs = trainer.model.model(X_batch)
                losses = trainer.loss_function(outputs, y_batch)
                mxnet.nd.waitall()
                autograd.backward(losses)
            trainer.optimizer.step(batch_size)
            for loss in losses:
                trainer.loss += loss.asnumpy()[0]
            if i % summarysteps == 0:
                print(trainer.global_step, i, 'loss:', trainer.loss / (i+1))
                if trainer.summarywriter:
                    trainer.summarywriter.add_scalar(
                        tag='loss', value=trainer.loss / (i+1), global_step=trainer.global_step)
                    trainer.summarywriter.add_image(
                        "image", (X_batch[0]/255.0), global_step=trainer.global_step)
                    trainer.summarywriter.add_image(
                        "mask", (y_batch[0]/255.0), global_step=trainer.global_step)
                    # with autograd.predict_mode():
                    #    outputs2 = trainer.model.model.module(X_batch.as_in_context(self.ctx))
                    output, _ = outputs[0]
                    #    print(output, outputs[0])
                    predict = mxnet.nd.squeeze(
                        mxnet.nd.argmax(output, 1)).asnumpy().clip(0, 1)
                    trainer.summarywriter.add_image(
                        "predicted", (predict), global_step=trainer.global_step)
        print('train on gluoncv backend')

    def get_summary_writer(self, logdir='results/'):
        return SummaryWriter(logdir=logdir)

    def predict(self, predictor, img):
        img_batch = batchify.Stack()([img])
        return self.batch_predict(predictor, img_batch)

    def batch_predict(self, predictor, img_batch):
        model = predictor.model.model
        with autograd.predict_mode():
            outputs = model(img_batch.as_in_context(self.ctx))
            output, _ = outputs
        predict = mxnet.nd.squeeze(
            mxnet.nd.argmax(output, 1)).asnumpy().clip(0, 1)
        return predict.astype(np.float32)
