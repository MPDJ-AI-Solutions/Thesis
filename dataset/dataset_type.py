from enum import Enum


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    UNITTEST = "ut"
    EASY_TRAIN = "train_easy"

    def describe(self):
        return f"Dataset is in state: {self.name}"

    def get_folder_name(self):
        return f"STARCOP_{self.value}"
