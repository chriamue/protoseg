#!/usr/bin/env python3

import argparse
import os
import sys
import cv2
import numpy as np

from protoseg import Augmentation
from protoseg import Config
from protoseg import DataLoader
from protoseg import Model
from protoseg import Trainer
from protoseg import Report
from protoseg import backends

resultspath = 'results/'

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--config', help="Path to config file.")
    
    args, _ = parser.parse_known_args()

    configfile = args.config or {'run1':{}}
    configs = Config(configfile)
    for run in configs:
        print("Run: ", run)
        resultpath = os.path.join(resultspath, run)
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        configs.save(resultpath + '/config.yml')
        # get config for current run
        config = configs.get()
        # set backend
        backends.set_backend(config['backend'])
        # summary
        summarywriter = backends.backend().get_summary_writer(logdir=resultpath)
        # Load Model
        modelfile = os.path.join('results/', run, 'model.checkpoint')
        model = Model(config, modelfile)
        # augmentation
        augmentation = Augmentation(config=config)
        # data loader
        dataloader = DataLoader(config=config, mode='train', augmentation=augmentation)
        # validation data loader
        valdataloader = DataLoader(config=config, mode='val')
        trainer = Trainer(config, model, dataloader, valdataloader=valdataloader, summarywriter=summarywriter)
        trainer.train()
    
    report = Report(configs, resultspath)
    report.generate()
    sys.exit(0)


if __name__ == '__main__':
    main()