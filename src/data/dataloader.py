import os
from src.config import DataConfig
from src.constants import FileNames
import torch
from data.data_loading import WeatherDataset


def load_train_and_test_datasets(data_path: str, data_config: DataConfig):

    feats_flattened = data_config.feats_flattened

    num_longitudes = data_config.num_longitudes
    num_latitudes = data_config.num_latitudes
    num_features = data_config.num_features
    obs_window = data_config.obs_window
    pred_window = data_config.pred_window

    num_features_used = data_config.num_features_used
    obs_window_used = data_config.obs_window_used
    pred_window_used = data_config.pred_window_used
    want_feats_flattened = data_config.want_feats_flattened

    
    # assertions to test data_config
    assert num_features_used <= num_features
    assert obs_window_used <= obs_window
    assert pred_window_used <= pred_window


    X_train_path = os.path.join(data_path, FileNames.TRAIN_X)
    y_train_path = os.path.join(data_path, FileNames.TRAIN_Y)

    X_test_path = os.path.join(data_path, FileNames.TEST_X)
    y_test_path = os.path.join(data_path, FileNames.TEST_Y)

    X_train = torch.load(X_train_path)
    y_train = torch.load(y_train_path)
    X_test = torch.load(X_test_path)
    y_test = torch.load(y_test_path)

    print("X_train from dataset shape: ", X_train.shape)
    print("y_train from dataset shape: ", y_train.shape)


    grid_dimension_size = num_longitudes * num_latitudes

    # handle the dataset differently if it is already flattened
    if feats_flattened:
        _, LONG, LAT, X_F = X_train.shape
        _, _, _, Y_F = y_train.shape

        #more assertions
        assert LONG == num_longitudes
        assert LAT == num_latitudes
        assert X_F == num_features * obs_window
        assert Y_F == num_features * pred_window

    # if the dataset is not flattened - the features and windows are in separate dimensions
    else:
        _, LONG, LAT, OBS, X_F = X_train.shape
        _, _, _, PRED, Y_F = y_train.shape

        #more assertions
        assert LONG == num_longitudes
        assert LAT == num_latitudes
        assert X_F == num_features
        assert OBS == obs_window
        assert Y_F == X_F
        assert PRED == pred_window



    # reshape the data so we can filter out the features we want to use
    X_train = X_train.reshape(-1, grid_dimension_size, obs_window, num_features)
    y_train = y_train.reshape(-1, grid_dimension_size, pred_window, num_features)
    X_test = X_test.reshape(-1, grid_dimension_size, obs_window, num_features)
    y_test = y_test.reshape(-1, grid_dimension_size, pred_window, num_features)

    # filter out the features we want to use
    X_train = X_train[:, :, (obs_window - obs_window_used):, :num_features_used]
    y_train = y_train[:, :, (pred_window - pred_window_used):, :num_features_used]
    X_test = X_test[:, :, (obs_window - obs_window_used):, :num_features_used]
    y_test = y_test[:, :, (pred_window - pred_window_used):, :num_features_used]

    # reshape the data back to the original shape if we want it that way
    if want_feats_flattened:
        X_train = X_train.reshape(-1, grid_dimension_size, obs_window_used * num_features_used)
        y_train = y_train.reshape(-1, grid_dimension_size, pred_window_used * num_features_used)
        X_test = X_test.reshape(-1, grid_dimension_size, obs_window_used * num_features_used)
        y_test = y_test.reshape(-1, grid_dimension_size, pred_window_used * num_features_used)



    train_dataset = WeatherDataset(X=X_train, y=y_train)
    test_dataset = WeatherDataset(X=X_test, y=y_test)

    return train_dataset, test_dataset
