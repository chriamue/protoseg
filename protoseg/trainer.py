
from .config import Config
from .dataloader import DataLoader
from .metric import Metric
from . import backends

class Trainer():

    epoch = 0
    global_step = 0
    loss = 0.0

    
    def before_epoch(self):
        print('starting epoch:', self.epoch)
        self.loss = 0.0

    def after_epoch(self):
        print('epoch finished. loss:', self.loss)
        
        backends.backend().save_model(self.model)



    before_epoch_callback = before_epoch
    after_epoch_callback = after_epoch

    def __init__(self, config = None, model = None, dataloader = None, valdataloader=None, summarywriter=None):
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.valdataloader = valdataloader
        self.summarywriter = summarywriter
        assert(config)
        assert(model)
        assert(dataloader)
        self.metric = Metric(self.config, self.summarywriter)
        backends.backend().init_trainer(self)

    def print_config(self):
        print('Epochs:' , self.config['epochs'])
        print('Learning rate:', self.config['learn_rate'])
        print('Batch size:', self.config['batch_size'])

    def train(self):
        self.global_step = 0
        self.print_config()
        for epoch in range(self.config['epochs']):
            self.epoch = epoch
            if self.before_epoch_callback:
                self.before_epoch_callback()
            
            backends.backend().train_epoch(self)

            if self.valdataloader:
                backends.backend().validate_epoch(self)

            if self.after_epoch_callback:
                self.after_epoch_callback()
