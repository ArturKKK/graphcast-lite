import xarray as xr
import numpy as np
import torch

from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader



def get_xarray_dataset(url: str, chunk_size: int = 48):
    """
    Get the dataset from the given url

    params:
    url: 
        string: the url of the dataset

    returns:
    dataset: 
        xarray.Dataset: the dataset

    example:
    dataset = get_dataset('gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr')
    """
    era5 = xr.open_zarr(
        url,
        chunks={'time': chunk_size},
        consolidated=True,
    )
    print("Model surface dataset size {} GiB".format(era5.nbytes/(1024**3)))
    return era5

def get_selected_variables(dataset: xr.Dataset, variables: list):
    """
    Get the selected variables from the dataset

    params:
    dataset: 
        xarray.Dataset: the dataset
    variables: 
        list: the list of variables to be selected

    returns:
    dataset: 
        xarray.Dataset: the dataset with the selected variables

    example:
    dataset = get_selected_variables(dataset, ['10m_wind_speed', '2m_temperature'])
    """
    dataset = dataset[variables]
    return dataset

def get_selected_time_range(dataset: xr.Dataset, start_time: str, end_time: str):
    """
    Get the selected time range from the dataset

    params:
    dataset: 
        xarray.Dataset: the dataset
    start_time: 
        string: the start time of the time range
    end_time: 
        string: the end time of the time range

    returns:
    dataset: 
        xarray.Dataset: the dataset with the selected time range

    example:
    dataset = get_selected_time_range(dataset, '2010-01-01', '2010-12-31')
    """
    dataset = dataset.sel(time=slice(start_time, end_time))
    return dataset


def write_dataset_to_netcdf(dataset: xr.Dataset, file_path: str):
    """
    Write the dataset to the netcdf file

    params:
    dataset: 
        xarray.Dataset: the dataset
    file_path: 
        string: the file path to save the dataset

    example:
    write_dataset_to_netcdf(dataset, 'data/era5_2010.nc')
    """
    dataset.to_netcdf(file_path)


def read_dataset_from_netcdf(file_path: str, chunk_size: int = 10):
    """
    Read the dataset from the netcdf file

    params:
    file_path: 
        string: the file path to read the dataset

    returns:
    dataset: 
        xarray.Dataset: the dataset

    example:
    dataset = read_dataset_from_netcdf('data/era5_2010.nc')
    """
    dataset = xr.open_dataset(file_path, chunks={'time': chunk_size})
    return dataset





class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    




def get_weather_dataset_from_array(
        dataset: xr.DataArray, 
        obs_window: int, 
        pred_window: int, 
        overlap: bool = False, 
        group_by_time: bool = True,
        test_size: float = 0.2,
        return_tensors: bool = True
    ):
    """
    From the dataset, get a pytorch Dataset instance

    params:
    dataset: 
        xarray.DataArray: the dataset
    obs_window: 
        int: the number of time steps in the observation window
    pred_window: 
        int: the number of time steps in the prediction window
    overlap: 
        bool: whether the windows overlap or not
    group_by_time:
        bool: whether we want to concat as [x_(t-n), ..., x_t] or have [f_1, ..., f_n] where f_j is the jth feature but with data of all time stamps

    returns:
    train_dataset:
        WeatherDataset: the training dataset
    test_dataset:
        WeatherDataset: the testing dataset

    example:
    data, targets = get_data_targets_from_dataset(dataset, obs_window=5, pred_window=1)
    """
    window = obs_window + pred_window
    # Apply the rolling function
    rolled_dataset = dataset.rolling(time=window, center=False).construct('window')

    # rolled_dataset.stack(space=['window', 'variable'])
    rolled_dataset = rolled_dataset.assign_coords(window=np.arange(1 - obs_window, pred_window + 1, dtype=np.int8))


    # Drop the NaNs introduced by rolling
    rolled_dataset = rolled_dataset.dropna(dim='time')

    if not overlap:
        # Manually slice the data to get non-overlapping windows
        rolled_dataset = rolled_dataset.isel(time=slice(0, None, window))

    # Get the data and targets
    X = rolled_dataset.isel(window=slice(obs_window))
    y = rolled_dataset.isel(window=slice(obs_window, None))

    print('Lag features created successfully')


    # Make array 2d
    X = X.stack(space=['longitude', 'latitude', 'window', 'variable'])
    y = y.stack(space=['longitude', 'latitude', 'window', 'variable'])



    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=42)

    print('Train-test split done')


    grouping = ['window', 'variable'] if group_by_time else ['variable', 'window']

    # Scaling
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()


    # Since this is regression, we must also scale the y
    X_train_scaled = X_scaler.fit_transform(X_train).unstack().stack(lag_features=grouping)
    y_train_scaled = y_scaler.fit_transform(y_train).unstack().stack(lag_features=grouping)

    X_test_scaled = X_scaler.transform(X_test).unstack().stack(lag_features=grouping)
    y_test_scaled = y_scaler.transform(y_test).unstack().stack(lag_features=grouping)

    print('Scaling done')

    X_train_scaled = torch.tensor(X_train_scaled.values, dtype=torch.float32)
    y_train_scaled = torch.tensor(y_train_scaled.values, dtype=torch.float32)
    X_test_scaled = torch.tensor(X_test_scaled.values, dtype=torch.float32)
    y_test_scaled = torch.tensor(y_test_scaled.values, dtype=torch.float32)

    print('Converted to tensors')

    if return_tensors:
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled
    
    else:
        # Create dataset instances
        train_dataset = WeatherDataset(X_train_scaled, y_train_scaled)
        test_dataset = WeatherDataset(X_test_scaled, y_test_scaled)

        return train_dataset, test_dataset










def get_dataset(url: str, asTensors: bool = True):
    """
    Get the dataset from the given url

    params:
    url: 
        string: the url of the dataset

    returns:
    returns:
    train_dataset:
        WeatherDataset: the training dataset
    test_dataset:
        WeatherDataset: the testing dataset

    example:
    dataset = get_dataset('gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr')
    """
    era5 = get_xarray_dataset(url, chunk_size=48)
    era5 = era5[['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']]
    era5 = era5.sel(time=slice('2010-01-01', '2010-12-31'))
    era5 = era5.to_array()

    if asTensors:
        X_train, y_train, X_test, y_test = get_weather_dataset_from_array(era5, obs_window=4, pred_window=1, overlap=False, group_by_time=True, test_size=0.2, return_tensors=asTensors)
        return X_train, y_train, X_test, y_test
    else:
        train_dataset, test_dataset = get_weather_dataset_from_array(era5, obs_window=4, pred_window=1, overlap=False, group_by_time=True, test_size=0.2, return_tensors=asTensors)
        return train_dataset, test_dataset