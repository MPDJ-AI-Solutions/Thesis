import os
import pickle
from typing import Tuple

import pandas as pd

from torch import nn
from datetime import datetime
from models.Tools.Measures.model_type import ModelType
from models.Tools.model_wrapper import ModelWrapper


class ModelFilesHandler:
    """
    Model saving module, which implements functionality of saving artificial intelligence model and
    its details such as date saved, metrics etc.
    """

    def __init__(self, base_dir:str = ""):
        self.models_directory_path = os.path.join(base_dir, "trained_models")
        self.csv_path = os.path.join(base_dir, "trained_models/models_descriptions.csv")

        os.makedirs(self.models_directory_path, exist_ok=True)
        if not os.path.exists(self.csv_path):
            self._create_description_file()

    def _create_description_file(self):
        # Define the columns for the description file
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

    def save_model(self, model: nn.Module, model_type: ModelType, metrics: pd.DataFrame, epoch: int):
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

    def load_model(self, csv_id: int = -1, file_name: str = "") -> Tuple[nn.Module, ModelType, pd.DataFrame, int]:
        assert (csv_id != -1 and csv_id >= 0) or file_name != "", "You need to specify id or file name."
        assert not ((csv_id >= 0) and not (file_name != "")), "You cannot specify id and file name in the same time."

        if file_name != "":
            wrapped_model = self._load_model_from_path(file_name)
            return wrapped_model.model, wrapped_model.model_type, wrapped_model.metrics, wrapped_model.epoch
        else:
            model_path = self._get_file_path(csv_id)
            wrapped_model = self._load_model_from_path(model_path)
            return wrapped_model.model, wrapped_model.model_type, wrapped_model.metrics, wrapped_model.epoch

    @staticmethod
    def _load_model_from_path(file_name) -> ModelWrapper:
        with open(file_name, mode='rb') as file:
            wrapped_model = pickle.load(file)

        return wrapped_model

    def _get_file_path(self, csv_id: int) -> str:
        csv = pd.read_csv(self.csv_path)
        return csv.iloc[csv_id]['file_name']
