from src.config import DatasetNames


class DatasetMetadata:
    def __init__(
        self,
        flattened: bool,
        num_latitudes: int,
        num_longitudes: int,
        num_features: int,
        obs_window: int,
        pred_window,
    ):
        self.flattened = flattened
        self.num_latitudes = num_latitudes
        self.num_longitudes = num_longitudes
        self.num_features = num_features
        self.obs_window = obs_window
        self.pred_window = pred_window


def get_dataset_metadata(dataset_name: DatasetNames) -> DatasetMetadata:
    if dataset_name == DatasetNames._64x32_10f_5y_3obs:
        return DatasetMetadata(
            flattened=True,
            num_latitudes=32,
            num_longitudes=64,
            num_features=10,
            obs_window=3,
            pred_window=1,
        )
    elif dataset_name == DatasetNames._64x32_33f_5y_5obs_uns:
        return DatasetMetadata(
            flattened=False,
            num_latitudes=32,
            num_longitudes=64,
            num_features=33,
            obs_window=5,
            pred_window=1,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
