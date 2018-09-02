from __future__ import absolute_import
name = "protoseg"

from . import backends
from .augmentation import Augmentation
from .config import Config
from .dataloader import DataLoader
from .trainer import Trainer
from .model import Model

__version__ = '0.0.1'