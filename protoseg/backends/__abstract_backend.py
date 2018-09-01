
class AbstractBackend():
    def dataloader_format(self, img, mask):
        return img, mask

    def load_model(self, config, modelfile):
        pass
    
    def save_model(self, model):
        pass

    def init_trainer(self, trainer):
        pass

    def train_epoch(self, trainer):
        pass

    def predict(self):
        pass
