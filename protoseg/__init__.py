from __future__ import absolute_import
name = "protoseg"

from . import backends
from .augmentation import Augmentation
from .config import Config
from .dataloader import DataLoader
from .metric import Metric
from .model import Model
from .predictor import Predictor
from .report import Report
from .trainer import Trainer

__version__ = '0.0.1'