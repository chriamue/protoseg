#!/usr/bin/env python

import os
import sys

from protoseg import Augmentation
from protoseg import Config
from protoseg import DataLoader
from protoseg import Model
from protoseg import Trainer
from protoseg import Report
from protoseg import HyperParamOptimizer
from protoseg import backends

resultspath = 'results/'

def help():
    return "Config file parameter missing. Run like: python train.py /path/to/config.yml 100, where 100 is max_evals."


if __name__ == "__main__":
    max_evals = 10
    if len(sys.argv) < 2:
        print(help())
        sys.exit(1)
    configs = Config(sys.argv[1])
    if len(sys.argv) > 2:
        max_evals = int(sys.argv[2])

    report = Report(configs, resultspath)
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
        modelfile = os.path.join(resultpath, 'model.checkpoint')
        model = Model(config, modelfile)
        # augmentation
        augmentation = Augmentation(config=config)
        # data loader
        dataloader = DataLoader(config=config, mode='train', augmentation=augmentation)
        # validation data loader
        valdataloader = DataLoader(config=config, mode='val')
        trainer = Trainer(config, model, dataloader, valdataloader=valdataloader, summarywriter=summarywriter)
        hyperoptimizer = HyperParamOptimizer(trainer)
        best = hyperoptimizer(max_evals)
        print(config)
        print(best)


        report.hyperparamopt(config, hyperoptimizer, resultpath)

    
    sys.exit(0)