import os
from src.constants import FileNames
import torch
from data.data_loading import WeatherDataset


def load_train_and_test_datasets(data_path: str):
    x_train_path = os.path.join(data_path, FileNames.TRAIN_X)
    y_train_path = os.path.join(data_path, FileNames.TRAIN_Y)

    x_test_path = os.path.join(data_path, FileNames.TEST_X)
    y_test_path = os.path.join(data_path, FileNames.TEST_Y)

    x_train = torch.load(x_train_path).view(-1, 64*32, 12)
    y_train = torch.load(y_train_path).view(-1, 64*32, 3)
    x_test = torch.load(x_test_path).view(-1, 64*32, 12)
    y_test = torch.load(y_test_path).view(-1, 64*32, 3)

    train_dataset = WeatherDataset(X=x_train, y=y_train)
    test_dataset = WeatherDataset(X=x_test, y=y_test)

    return train_dataset, test_dataset
