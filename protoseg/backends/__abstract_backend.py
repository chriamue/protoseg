
class AbstractBackend():
    def dataloader_format(self, img, mask):
        return img, mask
    
    def train_epoch(self, trainer):
        pass
    def predict(self):
        pass