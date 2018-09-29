import numpy as np
from . import backends
from importlib import import_module


class Predictor():

    def __init__(self, model, config={}, backend=backends.backend()):
        self.model = model
        self.config = config
        self.backend = backend
        assert(model)

        self.postprocessors = []
        postprocessors = self.config.get('postprocessors')
        if postprocessors:
            print('___ loading postprocessors ___')
            for f in postprocessors:
                full_function = list(f.keys())[0]
                module_name, function_name = full_function.rsplit('.', 1)
                parameters = f[full_function]
                print(module_name, function_name, parameters)
                mod = import_module(module_name)
                met = getattr(mod, function_name)
                self.postprocessors.append(
                    {'function': met, 'parameters': parameters})

    def postprocessing(self, img):
        for f in self.postprocessors:
            if type(f['parameters']) is list:
                img = f['function'](img, *f['parameters'])
            else:
                img = f['function'](img, **f['parameters'])
        return img

    def predict(self, img):
        prediction = self.backend.predict(self, img)
        return self.postprocessing(prediction)

    def batch_predict(self, img_batch):
        prediction = self.backend.batch_predict(self, img_batch)
        return self.postprocessing(prediction)
