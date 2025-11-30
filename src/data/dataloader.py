import os
from src.config import DataConfig
from src.constants import FileNames
import torch
from src.data.data_configs import DatasetMetadata, get_dataset_metadata
from torch.utils.data import Dataset
import numpy as np  # FIX: для чтения coords.npz

# При предикте для НСК у нас сетка получилась 61 на 41, а обучались мы на 64 на 32. Это проверяется тут , поэтому предикт не запускался
# Тут сделан фикс для этой проверки

class WeatherDataset(Dataset):
    """Simple tensor-based dataset used for training."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_train_and_test_datasets(data_path: str, data_config: DataConfig):

    dataset_metadata: DatasetMetadata = get_dataset_metadata(
        dataset_name=data_config.dataset_name
    )

    # --- то, что ожидалось из train-конфига ---
    feats_flattened_cfg = dataset_metadata.flattened
    num_longitudes_cfg = dataset_metadata.num_longitudes
    num_latitudes_cfg  = dataset_metadata.num_latitudes
    num_features       = dataset_metadata.num_features
    obs_window         = dataset_metadata.obs_window
    pred_window        = dataset_metadata.pred_window

    num_features_used  = data_config.num_features_used
    obs_window_used    = data_config.obs_window_used
    pred_window_used   = data_config.pred_window_used
    want_feats_flattened = data_config.want_feats_flattened

    # assertions to test data_config (оставляем — они про каналы/окна, не про размер решётки)
    assert num_features_used <= num_features
    assert obs_window_used   <= obs_window
    assert pred_window_used  <= pred_window

    X_train_path = os.path.join(data_path, FileNames.TRAIN_X)
    y_train_path = os.path.join(data_path, FileNames.TRAIN_Y)
    X_test_path  = os.path.join(data_path, FileNames.TEST_X)
    y_test_path  = os.path.join(data_path, FileNames.TEST_Y)

    X_train = torch.load(X_train_path)
    y_train = torch.load(y_train_path)
    X_test  = torch.load(X_test_path)
    y_test  = torch.load(y_test_path)

    # ----------------------------------------------------------------------
    # FIX(1): определяем фактическую форму датасета и снимаем жёсткую привязку
    # к num_longitudes/num_latitudes из конфигурации (64x32). Берём из файла.
    # Также автоматически детектим "flattened" по рангу тензора.
    # ----------------------------------------------------------------------
    if X_train.ndim == 4:  # [N, LONG, LAT, obs*F]
        feats_flattened = True
        _, LONG, LAT, X_F = X_train.shape
        _, _, _, Y_F = y_train.shape

        # Проверяем только соответствие каналов/окон
        assert X_F == num_features * obs_window
        assert Y_F == num_features * pred_window

    elif X_train.ndim == 5:  # [N, LONG, LAT, OBS, F]
        feats_flattened = False
        _, LONG, LAT, OBS, X_F = X_train.shape
        _, _, _, PRED, Y_F = y_train.shape

        assert X_F == num_features
        assert Y_F == X_F
        assert OBS == obs_window
        assert PRED == pred_window
    else:
        raise RuntimeError(f"Unexpected X_train.ndim={X_train.ndim}")

    # Если размеры из файла не совпали с «ожидаемыми», просто переопределим метаданные
    if (LONG != num_longitudes_cfg) or (LAT != num_latitudes_cfg):
        # print(f"[INFO] override grid shape from data: {num_longitudes_cfg}x{num_latitudes_cfg} -> {LONG}x{LAT}")
        dataset_metadata.num_longitudes = int(LONG)
        dataset_metadata.num_latitudes  = int(LAT)

    # FIX(2): если есть coords.npz, используем реальные оси; иначе — равномерные.
    coords_npz = os.path.join(data_path, "coords.npz")
    if os.path.exists(coords_npz):
        cz   = np.load(coords_npz)
        lons = cz["longitude"].astype(np.float32)  # длина = LONG
        lats = cz["latitude"].astype(np.float32)   # длина = LAT
    else:
        # равномерная сетка (как было в predict.linspace_lats_lons)
        lats = np.linspace(-90, 90, LAT, endpoint=True).astype(np.float32)
        lons = np.linspace(0, 360, LONG, endpoint=False).astype(np.float32)

    # Если в DatasetMetadata есть поле cordinates — положим туда; если нет, добавим динамически.
    try:
        dataset_metadata.cordinates = (lats, lons)  # WeatherPrediction берёт их дальше
    except Exception:
        # dataclass может не запрещать новые атрибуты — обычно ок
        setattr(dataset_metadata, "cordinates", (lats, lons))

    grid_dimension_size = LONG * LAT

    # ----------------------------------------------------------------------
    # Дальше логика та же, только используем фактический feats_flattened
    # (а не то, что в конфиге набора). Это защитит от несовпадений.
    # ----------------------------------------------------------------------
    if feats_flattened:
        # X: [N, LONG, LAT, obs*F] -> [N, G, obs, F]
        X_train = X_train.view(-1, grid_dimension_size, obs_window, num_features)
        y_train = y_train.view(-1, grid_dimension_size, pred_window, num_features)
        X_test  = X_test.view(-1,  grid_dimension_size, obs_window, num_features)
        y_test  = y_test.view(-1,  grid_dimension_size, pred_window, num_features)
    else:
        # Уже в форме [N, LONG, LAT, OBS, F] -> [N, G, OBS, F]
        X_train = X_train.view(-1, grid_dimension_size, obs_window, num_features)
        y_train = y_train.view(-1, grid_dimension_size, pred_window, num_features)
        X_test  = X_test.view(-1,  grid_dimension_size, obs_window, num_features)
        y_test  = y_test.view(-1,  grid_dimension_size, pred_window, num_features)

    # фильтруем по числу используемых фич/окон
    X_train = X_train[:, :, (obs_window - obs_window_used):, :num_features_used]
    y_train = y_train[:, :, (pred_window - pred_window_used):, :num_features_used]
    X_test  = X_test[:,  :, (obs_window - obs_window_used):, :num_features_used]
    y_test  = y_test[:,  :, (pred_window - pred_window_used):, :num_features_used]

    # возвращаем в плоский вид, если так нужно модели
    if want_feats_flattened:
        X_train = X_train.reshape(-1, grid_dimension_size, obs_window_used * num_features_used)
        y_train = y_train.reshape(-1, grid_dimension_size, pred_window_used * num_features_used)
        X_test  = X_test.reshape(-1,  grid_dimension_size, obs_window_used * num_features_used)
        y_test  = y_test.reshape(-1,  grid_dimension_size, pred_window_used * num_features_used)

    # валидацию берём из первой половины теста (как раньше)
    test_size = X_test.shape[0]
    val_size  = test_size // 2
    X_val = X_test[:val_size]
    y_val = y_test[:val_size]
    X_test = X_test[val_size:]
    y_test = y_test[val_size:]

    train_dataset = WeatherDataset(X=X_train, y=y_train)
    val_dataset   = WeatherDataset(X=X_val,   y=y_val)
    test_dataset  = WeatherDataset(X=X_test,  y=y_test)

    return train_dataset, val_dataset, test_dataset, dataset_metadata
