from .measure_tool import MeasureTool
from .model_type import ModelType
from .measure_tool_cnn import MeasureToolCNN
from .measure_tool_transformer import MeasureToolTransformer

class MeasureToolFactory:
    """
    Static factory used to create measure tool objects, accordingly to type of the model measured.
    """
    @staticmethod
    def get_measure_tool(model_type: ModelType) -> MeasureTool:
        match model_type:
            case ModelType.CNN:
                return MeasureToolCNN()
            case ModelType.TRANSFORMER:
                return MeasureToolTransformer()