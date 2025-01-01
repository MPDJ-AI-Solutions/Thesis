import pandas as pd

from torch import nn
from measures.model_type import ModelType


class ModelWrapper(object):
    """
    A wrapper class for handling model-related operations and metadata.

    Attributes:
        model (nn.Module): The neural network model.
        model_type (ModelType): The type of the model.
        epoch (int): The current epoch number.
        metrics (pd.DataFrame): A DataFrame containing the model's performance metrics.
        date (str): The date when the model was created or last modified.

    Args:
        model (nn.Module): The neural network model to be wrapped.
        model_type (ModelType): The type of the model.
        metrics (pd.DataFrame): A DataFrame containing the model's performance metrics.
        epoch (int): The current epoch number.
        date (str): The date when the model was created or last modified.
    """
    def __init__(self, model: nn.Module, model_type: ModelType, metrics: pd.DataFrame, epoch: int, date: str):
        """
        Initializes the ModelWrapper instance.

        Args:
            model (nn.Module): The neural network model.
            model_type (ModelType): The type of the model.
            metrics (pd.DataFrame): DataFrame containing the metrics.
            epoch (int): The current epoch number.
            date (str): The date of the model creation or training.
        """
        self.model = model
        self.model_type = model_type
        self.epoch = epoch
        self.metrics = metrics
        self.date = date
