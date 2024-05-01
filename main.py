import pandas as pd
from functions import *
import glob

# Setup catalogs paths
DATA_CATALOG = "./data/"
TEST_CATALOG = DATA_CATALOG + "STARCOP_test/STARCOP_test/"

# Load description of images
test_set_description = pd.read_csv(DATA_CATALOG + "test.csv")
train_set_description = pd.read_csv(DATA_CATALOG + "train.csv")
train_easy_description = pd.read_csv(DATA_CATALOG + "train_easy.csv")

photos_paths_list = [file for file in glob.glob(TEST_CATALOG + test_set_description['id'][256] + "/*.tif")]

for image_path in photos_paths_list:
    try:
        image_array = load_image(image_path)
        print(image_array)
    except Exception as exception:
        print('Cannot read image: ', image_path)

