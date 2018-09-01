
from .config import Config
from .dataloader import DataLoader
from . import backends

class Trainer():

    epoch = 0
    
    def before_epoch(self):
        print('starting epoch:', self.epoch)

    def after_epoch(self):
        print('epoch finished')

    before_epoch_callback = before_epoch
    after_epoch_callback = after_epoch

    def __init__(self, config = None, dataloader = None):
        self.config = config
        self.dataloader = dataloader
        assert(config)
        assert(dataloader)

    def print_config(self):
        print('Epochs:' , self.config['epochs'])
        print('Learning rate:', self.config['learn_rate'])
        print('Batch size:', self.config['batch_size'])

    def train(self):
        self.print_config()
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            if self.before_epoch_callback:
                self.before_epoch_callback()
            
            backends.backend().train_epoch(self)

            if self.after_epoch_callback:
                self.after_epoch_callback()
