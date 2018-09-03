
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
    
    def validate_epoch(self, trainer):
        pass

    def get_summary_writer(self, logdir='results/'):
        pass

    def predict(self, predictor, img):
        pass

    def batch_predict(self, predictor, img_batch):
        pass