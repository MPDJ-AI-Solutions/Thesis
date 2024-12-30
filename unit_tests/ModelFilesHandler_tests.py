import unittest
import os
from datetime import datetime
from unittest.mock import patch

import pandas as pd


from measures import ModelType
from files_handler import ModelFilesHandler
from models.Transformer.MethaneMapper import TransformerModel


class ModelFilesHandlerTests(unittest.TestCase):
    def setUp(self):
        self.csv_path  = "trained_models/models_descriptions.csv"
        self.model_dir = "trained_models"

    def tearDown(self):
        for file in os.listdir(self.model_dir):
            file_path = os.path.join(self.model_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def test_create_description_file(self):
        # Act
        mfh = ModelFilesHandler(base_dir="../unit_tests")

        # Assert
        self.assertTrue(os.path.exists(self.csv_path))

    def test_save_model(self):
        # Arrange
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2022, 1, 1, 12, 0, 0)

        mfh = ModelFilesHandler()
        model = TransformerModel()
        model_type = ModelType.CNN
        metrics = pd.DataFrame.from_dict({
            'index'       :[0],
            'tp'          :0,
            'fp'          :0,
            'fn'          :0,
            'tn'          :0,
            'precision'   :0,
            'sensitivity' :0,
            'specificity' :0,
            'npv'         :0,
            'fpr'         :0,
            'accuracy'    :0,
            'fscore'      :0,
            'iou'         :0,
            'mcc'         :0,
            'auc'         :0,
            'ci'          :0
        }, )
        epoch = 2

        model_name = f'model_{model_type.value}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pickle'
        model_path = os.path.join(self.model_dir, model_name)

        # Act
        mfh.save_model(model, model_type, metrics, epoch)

        # Assert
        self.assertTrue(os.path.exists(model_path))


if __name__ == '__main__':
    unittest.main()
