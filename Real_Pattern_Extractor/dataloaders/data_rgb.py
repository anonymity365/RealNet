import os
from .dataset_rgb import DataLoaderVal , DataLoaderTest, DataLoader_NoisyData, DataLoader_RPTC, read_data_new

def get_validation_data(rgb_dir, isVal=False):
    assert os.path.exists(rgb_dir)
    return read_data_new(rgb_dir, isVal=isVal)


def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)

def get_rgb_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoader_NoisyData(rgb_dir)

def get_rptc_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoader_RPTC(rgb_dir, None)