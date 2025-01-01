from enum import Enum


class DatasetType(Enum):
    """
    DatasetType is an enumeration that represents different types of datasets used in the project.
    Attributes:
        TRAIN (str): Represents the training dataset.
        TEST (str): Represents the testing dataset.
        UNITTEST (str): Represents the unit testing dataset.
        EASY_TRAIN (str): Represents an easier version of the training dataset.
    """
    TRAIN = "train"
    TEST = "test"
    UNITTEST = "ut"
    EASY_TRAIN = "train_easy"

    def describe(self):
        """
        Returns a string describing the current state of the dataset.

        Returns:
            str: A string indicating the state of the dataset.
        """
        return f"Dataset is in state: {self.name}"

    def get_folder_name(self):
        """
        Generates a folder name based on the value of the instance.

        Returns:
            str: The folder name in the format "STARCOP_<value>".
        """
        return f"STARCOP_{self.value}"
