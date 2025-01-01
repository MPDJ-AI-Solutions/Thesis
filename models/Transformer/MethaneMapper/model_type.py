from enum import Enum


class ModelType(Enum):
    """
    An enumeration representing different types of models.

    Attributes:
        CLASSIFICATION (str): Represents a classification model.
        SEGMENTATION (str): Represents a segmentation model.
        REGRESSION (str): Represents a regression model.
        ONLY_BBOX (str): Represents a model that only predicts bounding boxes.
    """
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"
    ONLY_BBOX = "only_bbox"
