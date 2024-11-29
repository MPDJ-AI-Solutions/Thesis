from enum import Enum


class ModelType(Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    REGRESSION = "regression"
    ONLY_BBOX = "only_bbox"
