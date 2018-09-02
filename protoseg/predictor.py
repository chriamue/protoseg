import numpy as np
from . import backends

class Predictor():

    def __init__(self, model):
        self.model = model
        assert(model)

    def predict(self, img):
        return backends.backend().predict(self, img)

    def batch_predict(self, img_batch):
        return backends.backend().batch_predict(self, img_batch)