import os
from src.config import DataConfig
from src.constants import FileNames
import torch
from data.data_loading import WeatherDataset


def load_train_and_test_datasets(data_path: str, data_config: DataConfig):
    x_train_path = os.path.join(data_path, FileNames.TRAIN_X)
    y_train_path = os.path.join(data_path, FileNames.TRAIN_Y)

    x_test_path = os.path.join(data_path, FileNames.TEST_X)
    y_test_path = os.path.join(data_path, FileNames.TEST_Y)

    grid_dimension_size = data_config.num_longitudes*data_config.num_latitudes

    x_train = torch.load(x_train_path).view(-1, grid_dimension_size, data_config.num_features*data_config.num_timesteps)
    y_train = torch.load(y_train_path).view(-1, grid_dimension_size, data_config.num_features)
    x_test = torch.load(x_test_path).view(-1, grid_dimension_size, data_config.num_features * data_config.num_timesteps)
    y_test = torch.load(y_test_path).view(-1, grid_dimension_size, data_config.num_features)

    train_dataset = WeatherDataset(X=x_train, y=y_train)
    test_dataset = WeatherDataset(X=x_test, y=y_test)

    return train_dataset, test_dataset
