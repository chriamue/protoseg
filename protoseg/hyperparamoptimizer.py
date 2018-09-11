import numpy as np
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval


class HyperParamOptimizer():
    trials = Trials()
    space = {
        'learn_rate': hp.loguniform('learn_rate',
                                    np.log(0.005),
                                    np.log(0.2)),
        'batch_size': hp.choice('batch_size', list(range(1, 3)))
    }
    epochs = 2

    def __init__(self, trainer):
        self.trainer = trainer

    def objective(self, params):
        self.trainer.model.load()
        self.trainer.init()
        for param in params:
            self.trainer.config[param] = params[param]
        self.trainer.train(self.epochs)
        return {'loss': self.trainer.loss, 'status': STATUS_OK}

    def after_epoch(self):
        pass

    def __call__(self, max_evals = 10):
        self.after_epoch_callback = self.trainer.after_epoch_callback

        self.trainer.after_epoch_callback = self.after_epoch
        best = fmin(self.objective, self.space, algo=tpe.suggest,
                    max_evals=max_evals, trials=self.trials)
        best_params = space_eval(self.space, best)
        self.trainer.after_epoch_callback = self.after_epoch_callback
        return best_params
