import os
import pickle
import pandas as pd

from torch import nn
from typing import Tuple
from datetime import datetime

from files_handler.model_wrapper import ModelWrapper
from measures.model_type import ModelType


class ModelFilesHandler:
    """
    Model saving module, which implements functionality of saving artificial intelligence model and
    its details such as date saved, metrics etc.
    """

    def __init__(self, base_dir:str = ""):
        """
        Initializes the ModelFilesHandler with the specified base directory.

        Args:
            base_dir (str): The base directory where the trained models and descriptions will be stored. Defaults to an empty string.
        Attributes:
            models_directory_path (str): The path to the directory where trained models are stored.
            csv_path (str): The path to the CSV file that contains models descriptions.
        Creates:
            The models directory if it does not exist.
            The models descriptions CSV file if it does not exist.
        """
        self.models_directory_path = os.path.join(base_dir, "trained_models")
        self.csv_path = os.path.join(base_dir, "trained_models/models_descriptions.csv")

        os.makedirs(self.models_directory_path, exist_ok=True)
        if not os.path.exists(self.csv_path):
            self._create_description_file()

    def _create_description_file(self):
        """
        Creates an empty description file with predefined columns.
        This method defines a set of columns relevant to model description,
        creates an empty pandas DataFrame with these columns, and saves it
        as a CSV file at the specified path.
        Columns:
            - model_type: Type of the model.
            - date: Date of the model creation or update.
            - file_name: Name of the file.
            - epoch: Epoch number.
            - tp: True positives.
            - fp: False positives.
            - fn: False negatives.
            - tn: True negatives.
            - precision: Precision metric.
            - sensitivity: Sensitivity metric.
            - specificity: Specificity metric.
            - npv: Negative predictive value.
            - fpr: False positive rate.
            - accuracy: Accuracy metric.
            - fscore: F-score metric.
            - iou: Intersection over Union metric.
            - mcc: Matthews correlation coefficient.
            - auc: Area under the curve.
            - ci: Confidence interval.
        Saves:
            An empty CSV file with the specified columns at `self.csv_path`.
        """
        columns = [
            'model_type', 'date', 'file_name', 'epoch',
            'tp', 'fp', 'fn', 'tn', 'precision', 'sensitivity',
            'specificity', 'npv', 'fpr', 'accuracy', 'fscore',
            'iou', 'mcc', 'auc', 'ci',
        ]

        # Create an empty DataFrame with the specified columns
        df = pd.DataFrame(columns=columns)

        # Save the empty DataFrame as a CSV file
        df.to_csv(self.csv_path, index=False)

    def save_raw_model(self, model: nn.Module):
        """
        Saves the given raw model to a file in the specified directory.
        The model is saved with a filename that includes the current date and time in the format:
        'model_raw_YYYY_MM_DD_HH_MM_SS.pickle'.

        Args:
            model (nn.Module): The neural network model to be saved.
        """
        current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f'model_raw_{current_date}.pickle'
        model_path = os.path.join(self.models_directory_path, file_name)

        with open(model_path, mode='wb') as file:
            pickle.dump(model, file)

    def save_model(self, model: nn.Module, model_type: ModelType, metrics: pd.DataFrame, epoch: int):
        """
        Saves the given model along with its metrics and epoch information to a file.
        The function performs the following steps:
        1. Generates a unique file name based on the model type and current date and time.
        2. Creates a ModelWrapper object containing the model, model type, metrics, epoch, and current date.
        3. Serializes the ModelWrapper object to a file using pickle.
        4. Appends the model information and metrics to a CSV file for record-keeping.

        Args:
            model (nn.Module): The model to be saved.
            model_type (ModelType): The type of the model.
            metrics (pd.DataFrame): The metrics associated with the model.
            epoch (int): The epoch number at which the model is being saved.
        Returns:
            str: The path to the saved model file.
        """
        current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f'model_{model_type.value}_{current_date}.pickle'
        model_path = os.path.join(self.models_directory_path, file_name)

        object_to_save = ModelWrapper(model, model_type, metrics, epoch, current_date)
        with open(model_path, mode='wb') as file:
            pickle.dump(object_to_save, file)

        new_row = {
            'model_type': [model_type.value],
            'date': [current_date],
            'file_name': [file_name],
            'epoch': [epoch],
        }

        model_info = pd.DataFrame.from_dict(new_row)
        final_row = pd.concat([model_info, metrics], axis=1)

        final_row.to_csv(self.csv_path, mode='a', index=False, header=False)

        return model_path

    def load_model(self, csv_id: int = -1, file_name: str = "") -> Tuple[nn.Module, ModelType, pd.DataFrame, int]:
        """
        Load a model from a specified CSV ID or file name.
        
        Args:
            csv_id (int, optional): The ID of the CSV file to load the model from. Default is -1.
            file_name (str, optional): The name of the file to load the model from. Default is an empty string.
        Returns:
            Tuple[nn.Module, ModelType, pd.DataFrame, int]: A tuple containing the loaded model, the model type,
            the metrics DataFrame, and the epoch number.
        Raises:
            AssertionError: If neither csv_id nor file_name is specified, or if both are specified at the same time.
        """
        assert (csv_id != -1 and csv_id >= 0) or file_name != "", "You need to specify id or file name."
        assert not ((csv_id >= 0) and not (file_name != "")), "You cannot specify id and file name in the same time."

        if file_name != "":
            wrapped_model = self._load_model_from_path(file_name)
            return wrapped_model.model, wrapped_model.model_type, wrapped_model.metrics, wrapped_model.epoch
        else:
            model_path = self._get_file_path(csv_id)
            wrapped_model = self._load_model_from_path(model_path)
            return wrapped_model.model, wrapped_model.model_type, wrapped_model.metrics, wrapped_model.epoch

    def load_raw_model(self, path):
        """
        Load a raw model from a specified file path.

        Args:
            path (str): The file path to the model file.
        Returns:
            object: The loaded model object.
        """
        with open(path, mode='rb') as file:
            model = pickle.load(file)
        return model


    @staticmethod
    def _load_model_from_path(file_name) -> ModelWrapper:
        """
        Load a model from a specified file path.

        Args:
            file_name (str): The path to the file containing the model.
        Returns:
            ModelWrapper: The loaded model wrapped in a ModelWrapper instance.
        """
        with open(file_name, mode='rb') as file:
            wrapped_model = pickle.load(file)

        return wrapped_model

    def _get_file_path(self, csv_id: int) -> str:
        """
        Retrieve the file path from a CSV file based on the given CSV ID.

        Args:
            csv_id (int): The ID of the CSV row to retrieve the file path from.
        Returns:
            str: The file path corresponding to the given CSV ID.
        """
        csv = pd.read_csv(self.csv_path)
        return csv.iloc[csv_id]['file_name']
