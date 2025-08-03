"""Download a tiny ERA5 sample and store it in the format expected by the
training pipeline.

The script pulls a few days of data from the public WeatherBench2 bucket and
creates train/test tensors.  Only three variables are used so that the data is
small and the model can train quickly.
"""

from pathlib import Path
import numpy as np
import torch
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_URL = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"
)


def _build_samples(arr: np.ndarray, obs: int, pred: int):
    """Create lagged samples from an array of shape
    (time, lon, lat, features)."""

    X_list, y_list = [], []
    total = arr.shape[0] - obs - pred + 1
    for i in range(total):
        X_list.append(arr[i : i + obs])
        y_list.append(arr[i + obs : i + obs + pred])

    X = np.stack(X_list)  # (samples, obs, lon, lat, feat)
    y = np.stack(y_list)
    return X, y


def main():
    out_dir = Path("data/datasets/demo_era5_small")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = xr.open_zarr(
            DATA_URL,
            consolidated=True,
            chunks={"time": 10},
            storage_options={"token": "anon", "asynchronous": False},
        )
        ds = ds[
            [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
            ]
        ]
        ds = ds.sel(time=slice("2010-01-01", "2010-01-10"))
        arr = ds.to_array().transpose("time", "longitude", "latitude", "variable").values
    except Exception as e:  # pragma: no cover - network failures
        print(f"Dataset download failed ({e}); using random data instead.")
        arr = np.random.randn(10, 64, 32, 3)

    obs_window, pred_window = 2, 1
    X, y = _build_samples(arr, obs_window, pred_window)

    samples, _, lon, lat, feat = X.shape
    # reshape to (samples, lon, lat, obs_window*feat)
    X = X.reshape(samples, obs_window, lon, lat, feat).transpose(0, 2, 3, 1, 4)
    X = X.reshape(samples, lon, lat, obs_window * feat)
    y = y.reshape(samples, pred_window, lon, lat, feat).transpose(0, 2, 3, 1, 4)
    y = y.reshape(samples, lon, lat, pred_window * feat)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_train_flat = X_train.reshape(-1, obs_window * feat)
    y_train_flat = y_train.reshape(-1, pred_window * feat)
    X_train_scaled = scaler_x.fit_transform(X_train_flat).reshape(X_train.shape)
    y_train_scaled = scaler_y.fit_transform(y_train_flat).reshape(y_train.shape)
    X_test_scaled = scaler_x.transform(X_test.reshape(-1, obs_window * feat)).reshape(
        X_test.shape
    )
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, pred_window * feat)).reshape(
        y_test.shape
    )

    torch.save(torch.tensor(X_train_scaled, dtype=torch.float32), out_dir / "X_train.pt")
    torch.save(torch.tensor(y_train_scaled, dtype=torch.float32), out_dir / "y_train.pt")
    torch.save(torch.tensor(X_test_scaled, dtype=torch.float32), out_dir / "X_test.pt")
    torch.save(torch.tensor(y_test_scaled, dtype=torch.float32), out_dir / "y_test.pt")

    print(f"Saved tensors to {out_dir}")


if __name__ == "__main__":
    main()

