from .config import Config
from . import backends


class Model():
    model = None

    def __init__(self, config, modelfile):
        self.config = config
        self.modelfile = modelfile
        self.load()

    def load(self):
        if self.model:
            del self.model
        self.model = backends.backend().load_model(self.config, self.modelfile)
