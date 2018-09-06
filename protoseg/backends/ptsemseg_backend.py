from __future__ import absolute_import
import os
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *
from ptsemseg.utils import convert_state_dict

from protoseg.backends import AbstractBackend
from protoseg.trainer import Trainer

from tensorboardX import SummaryWriter


class ptsemseg_backend(AbstractBackend):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self):
        AbstractBackend.__init__(self)

    def load_model(self, config, modelfile):
        model = get_model(config['backbone'],
                          config['classes']).to(self.device)
        if os.path.isfile(modelfile):
            print('loaded model from:', modelfile)
            state = convert_state_dict(torch.load(modelfile)["model_state"])
            model.load_state_dict(state)
        model = torch.nn.DataParallel(
            model, device_ids=range(torch.cuda.device_count()))
        return model

    def save_model(self, model):
        state = {
            "model_state": model.model.state_dict()
        }
        torch.save(state,
                   model.modelfile)
        print('saved model to:', model.modelfile)

    def init_trainer(self, trainer):
        if hasattr(trainer.model.model.module, "optimizer"):
            print("Using custom optimizer")
            optimizer = trainer.model.model.module.optimizer(
                trainer.model.model.model.parameters())
        else:
            trainer.optimizer = torch.optim.SGD(trainer.model.model.parameters(),
                                                lr=trainer.config['learn_rate'],
                                                momentum=0.9,
                                                weight_decay=0.0005)

        if hasattr(trainer.model.model.module, "loss"):
            print("Using custom loss")
            trainer.loss_function = trainer.model.model.module.loss
        else:
            trainer.loss_function = cross_entropy2d

    def dataloader_format(self, img, mask=None):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = np.transpose(img, axes=[2, 0, 1])
        if mask is None:
            return img.astype(np.float32)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        mask[mask > 0] = 1  # binary mask
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(mask.astype(np.int64))
        return img.astype(np.float32), mask.astype(np.int64)

    def train_epoch(self, trainer):
        batch_size = trainer.config['batch_size']
        summarysteps = trainer.config['summarysteps']
        epoch_loss = 0

        dataloader = data.DataLoader(
            trainer.dataloader, batch_size=batch_size, num_workers=min(batch_size, 8), shuffle=True
        )

        for (images, labels) in dataloader:
            trainer.global_step += 1
            trainer.model.model.train()
            images = images.to(self.device)
            labels = labels.to(self.device)

            trainer.optimizer.zero_grad()
            outputs = trainer.model.model(images)

            loss = trainer.loss_function(input=outputs, target=labels)

            loss.backward()
            trainer.optimizer.step()

            if trainer.global_step % summarysteps == 0:
                print('{0:.4f} --- loss: {1:.6f}'.format(trainer.global_step *
                                                         batch_size / len(trainer.dataloader), loss.item()))
                trainer.summarywriter.add_scalar(
                    'loss', loss.item(), global_step=trainer.global_step)
                trainer.summarywriter.add_image(
                    'image', images[0], global_step=trainer.global_step)
                trainer.summarywriter.add_image(
                    'mask', labels[0], global_step=trainer.global_step)
                pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
                trainer.summarywriter.add_image(
                    'predicted', pred, global_step=trainer.global_step)

    def validate_epoch(self, trainer):
        batch_size = trainer.config['batch_size']
        dataloader = data.DataLoader(
            trainer.valdataloader, batch_size=batch_size, num_workers=min(batch_size, 8), shuffle=True
        )
        for i, (X_batch, y_batch) in enumerate(dataloader):
            prediction = self.batch_predict(trainer, X_batch)
            trainer.metric(prediction, y_batch[0].numpy())
            trainer.summarywriter.add_image(
                "val_image", (X_batch[0]/255.0), global_step=trainer.epoch)
            trainer.summarywriter.add_image(
                "val_mask", (y_batch[0]), global_step=trainer.epoch)
            trainer.summarywriter.add_image(
                "val_predicted", (prediction), global_step=trainer.epoch)

    def get_summary_writer(self, logdir='results/'):
        return SummaryWriter(log_dir=logdir)

    def predict(self, predictor, img):
        img_batch = [img]
        return self.batch_predict(predictor, img_batch)

    def batch_predict(self, predictor, img_batch):
        model = predictor.model.model

        try:
            model = model.module
        except Exception:
            pass
        model.eval()
        images = img_batch.to(self.device)
        outputs = model(images)
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        return pred
