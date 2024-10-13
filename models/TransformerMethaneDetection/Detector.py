from torch import nn

from models.TransformerMethaneDetection.Backbone.backbone import Backbone
from models.TransformerMethaneDetection.Transformer.transformer import Transformer


class Detector(nn.Module):
    # TODO connect models
    def __init__(self):
        super(Detector, self).__init__()
