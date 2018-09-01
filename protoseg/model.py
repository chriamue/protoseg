from .config import Config
from . import backends

class Model():
    def __init__(self, config, modelfile):
        self.config = config
        self.modelfile = modelfile
        self.model = backends.backend().load_model(config, modelfile)
    