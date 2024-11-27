from enum import Enum


class ModelType(Enum):
    """
    Enum describing the type of model. CNN for convolutional neural network model.
    """
    CNN = "cnn"
    TRANSFORMER = "transformer"
    TRANSFORMER_CLASSIFIER = "transformer_classifier"