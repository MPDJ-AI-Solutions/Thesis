from enum import Enum


class ModelType(Enum):
    """
    Enum class representing different types of models.
    Attributes:
        CNN (str): Convolutional Neural Network model type.
        DETR (str): Detection Transformer model type.
        VIT (str): Vision Transformer model type.
        MethaneMapper (str): Methane Mapper model type.
        TRANSFORMER (str): Transformer model type for legacy compatibility.
        TRANSFORMER_CLASSIFIER (str): Transformer Classifier model type for legacy compatibility.
    """
    CNN = "cnn"
    DETR = "detr"
    VIT = "vit"
    MethaneMapper = "mm"

    TRANSFORMER = "transformer"
    TRANSFORMER_CLASSIFIER = "transformer_classifier"