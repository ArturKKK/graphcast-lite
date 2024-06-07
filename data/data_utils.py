import xarray as xr

def get_dataset(url: str, chunk_size: int = 48):
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




def get_data_targets_from_dataset(dataset: xr.DataArray, obs_window: int, pred_window: int, overlap: bool = True):
    """
    Get the data and targets from the dataset with variable observation and prediction windows.
    The data and targets are concatenated along the features dimension.

    params:
    dataset: 
        xarray.DataArray: the dataset
    obs_window: 
        int: the number of time steps in the observation window
    pred_window: 
        int: the number of time steps in the prediction window
    overlap: 
        bool: whether the windows overlap or not

    returns:
    data: 
        xarray.DataArray: the data
        shape: (num_samples, obs_window, lon, lat, features)
    targets: 
        xarray.DataArray: the targets
        shape: (num_samples, pred_window, lon, lat, features)

    example:
    data, targets = get_data_targets_from_dataset(dataset, obs_window=5, pred_window=1)
    """
    X_list = []
    y_list = []
    
    time_steps = dataset.time.shape[0]
    
    step = 1 if overlap else obs_window + pred_window
    
    for start in range(0, time_steps - obs_window - pred_window + 1, step):
        end_obs = start + obs_window
        end_pred = end_obs + pred_window
        
        X = dataset.isel(time=slice(start, end_obs)).stack(space=['longitude', 'latitude', 'variable'])
        y = dataset.isel(time=slice(end_obs, end_pred)).stack(space=['longitude', 'latitude', 'variable'])
        
        X_list.append(X)
        y_list.append(y)
    
    X_combined = xr.concat(X_list, dim='sample')
    y_combined = xr.concat(y_list, dim='sample')
    
    return X_combined, y_combined

