import pandas as pd
from torch import nn

from models.Tools.Measures.model_type import ModelType


class ModelWrapper(object):
    def __init__(self, model: nn.Module, model_type: ModelType, metrics: pd.DataFrame, epoch: int, date: str):
        self.model = model
        self.model_type = model_type
        self.epoch = epoch
        self.metrics = metrics
        self.date = date
