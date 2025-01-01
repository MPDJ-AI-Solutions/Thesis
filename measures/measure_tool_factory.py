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
        """
        Factory method to get the appropriate MeasureTool instance based on the model type.

        Args:
            model_type (ModelType): The type of the model for which the measure tool is required.

        Returns:
            MeasureTool: An instance of a MeasureTool subclass corresponding to the given model type.
        """
        match model_type:
            case ModelType.CNN:
                return MeasureToolCNN()
            case ModelType.TRANSFORMER:
                return MeasureToolTransformer()