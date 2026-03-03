"""Main entrypoint to run training for Weather Prediciton"""
import sys
import os
from src.constants import FileNames, FolderNames
from src.config import ExperimentConfig
from src.utils import load_from_json_file
from torch.utils.data import DataLoader
import torch
from src.models import WeatherPrediction
import numpy as np
from torch.optim import Adam
from src.train import train
from src.data.dataloader import load_train_and_test_datasets
from src.data.dataloader_chunked import load_chunked_datasets
from src.data.data_configs import DatasetMetadata
from src.config import DatasetNames
import random

CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


# Синхронизирует случайности (random, numpy, torch) для воспроизводимости.
def set_random_seeds(seed: int = 42):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


# Создаёт регулярную широтно-долготную сетку
# Затем инициализирует WeatherPrediction, передавая:
# координаты сетки,
# graph_config (как строим рёбра иерархии mesh, grid→mesh/mesh→grid),
# pipeline_config (архитектура encoder/processor/decoder),
# data_config (сколько фич/окон реально использовать),
# device.
def load_model_from_experiment_config(
    experiment_config: ExperimentConfig, device, dataset_metadata: DatasetMetadata,
    coordinates=None, region_bounds=None, mesh_buffer: float = 15.0,
    flat_grid: bool = False,
) -> WeatherPrediction:

    if coordinates is not None:
        lats, longs = coordinates
    else:
        lats = np.linspace(
            start=-90,
            stop=90,
            num=dataset_metadata.num_latitudes,
            endpoint=True,
        )
        longs = np.linspace(
            start=0,
            stop=360,
            num=dataset_metadata.num_longitudes,
            endpoint=False,
        )

    model = WeatherPrediction(
        cordinates=(lats, longs),
        graph_config=experiment_config.graph,
        pipeline_config=experiment_config.pipeline,
        data_config=experiment_config.data,
        device=device,
        region_bounds=region_bounds,
        mesh_buffer=mesh_buffer,
        flat_grid=flat_grid,
    )

    return model


def run_experiment(experiment_config: ExperimentConfig, results_save_dir: str,
                   resume: bool = False, pretrained_ckpt: str = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_random_seeds(seed=experiment_config.random_seed)

    # Загружает датасеты
    if experiment_config.data.dataset_name in (
        DatasetNames.wb2_512x256_19f_ar,
        DatasetNames.wb2_512x256_19f_ar_v2,
        DatasetNames.multires,
        DatasetNames.region_krsk_cds_19f,
    ):
        # Chunked dataloader для больших сеток (512×256) и мультирезолюционных
        # Для AR-обучения подаём max_ar_steps целевых кадров
        ar_target_steps = max(experiment_config.max_ar_steps, 1)
        
        if experiment_config.data.dataset_name == DatasetNames.multires:
            # Multires: data_dir задаётся через конфиг (dataset_name — просто маркер)
            # По умолчанию пробуем data/datasets/multires_*
            if experiment_config.data_dir:
                data_path = experiment_config.data_dir
            else:
                physical_dataset = str(experiment_config.data.dataset_name.value)
                data_path = os.path.join("data", "datasets", physical_dataset)
        elif hasattr(experiment_config, 'data_dir') and experiment_config.data_dir:
            # Любой датасет с явным data_dir в конфиге (region_krsk_cds и т.п.)
            data_path = experiment_config.data_dir
        else:
            # v2 использует тот же датасет, что и v1
            physical_dataset = "global_512x256_19f_2010-2021_07deg"
            data_path = os.path.join("data", "datasets", physical_dataset)
        train_dataset, val_dataset, test_dataset, dataset_metadata = (
            load_chunked_datasets(
                data_path=data_path,
                obs_window=experiment_config.data.obs_window_used,
                pred_steps=ar_target_steps,
                n_features=experiment_config.data.num_features_used,
            )
        )
    else:
        # Legacy: загрузка из .pt файлов (64×32)
        train_dataset, val_dataset, test_dataset, dataset_metadata = (
            load_train_and_test_datasets(
                data_path=os.path.join(
                    "data", "datasets", experiment_config.data.dataset_name
                ),
                data_config=experiment_config.data,
            )
        )

    # shuffle в DataLoader — это «перемешивать ли порядок сэмплов при формировании батчей».
    use_cuda = device.type == "cuda"
    loader_kwargs = dict(
        num_workers=4 if use_cuda else 0,
        pin_memory=True if use_cuda else False,
        persistent_workers=True if use_cuda else False,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=experiment_config.batch_size, shuffle=True,
        **loader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=experiment_config.batch_size, shuffle=False,
        **loader_kwargs,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=experiment_config.batch_size, shuffle=False,
        **loader_kwargs,
    )

    # --- Автоматическое определение region_bounds для обрезки mesh ---
    # Если координаты задают маленький регион (не глобальную сетку), вырезаем из
    # икосаэдра только ближайшие к области mesh-узлы: это примерно на порядок
    # уменьшает число mesh-вершин и делает GNN-процессор эффективным.
    real_coords = getattr(dataset_metadata, 'cordinates', None)
    flat_grid = getattr(dataset_metadata, 'flat_grid', False)
    region_bounds = None
    if real_coords is not None:
        lats_arr, lons_arr = real_coords
        lat_span = float(lats_arr.max() - lats_arr.min())
        lon_span = float(lons_arr.max() - lons_arr.min())
        # Считаем «региональным», если покрытие < 90° по обоим осям
        if lat_span < 90 and lon_span < 90:
            region_bounds = (
                float(lats_arr.min()), float(lats_arr.max()),
                float(lons_arr.min()), float(lons_arr.max()),
            )
            print(f"[region] Обнаружен региональный датасет "
                  f"lat=[{region_bounds[0]:.1f},{region_bounds[1]:.1f}] "
                  f"lon=[{region_bounds[2]:.1f},{region_bounds[3]:.1f}] → обрезаем mesh")

    model: WeatherPrediction = load_model_from_experiment_config(
        experiment_config=experiment_config,
        device=device,
        dataset_metadata=dataset_metadata,
        coordinates=real_coords,
        region_bounds=region_bounds,
        flat_grid=flat_grid,
    )

    model = model.to(device)

    # Загрузка pretrained весов (fine-tuning из другой модели)
    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        print(f"\n>>> Загружаю pretrained веса: {pretrained_ckpt}")
        state = torch.load(pretrained_ckpt, map_location=device)
        # strict=False — пропустить буферы, чьи размеры не совпадут
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        print(f"  Pretrained веса загружены.\n")

    # Создание оптимизатора — алгоритма, который меняет веса модели по градиентам, чтобы минимизировать loss.
    optimizer = Adam(params=model.parameters(), lr=experiment_config.learning_rate)

    # Путь к чекпоинту для возобновления
    checkpoint_path = os.path.join(results_save_dir, "checkpoint.pth") if resume else None

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimiser=optimizer,
        num_epochs=experiment_config.num_epochs,
        device=device,
        config=experiment_config,
        results_save_dir=results_save_dir,
        dataset_metadata=dataset_metadata,
        print_losses=True,
        wandb_log=experiment_config.wandb_log,
        resume_checkpoint=checkpoint_path,
    )

    return results


def main():
    experiment_directory = sys.argv[1]
    resume = "--resume" in sys.argv

    # --pretrained path/to/model.pth — загрузить веса другой модели для fine-tuning
    pretrained_ckpt = None
    if "--pretrained" in sys.argv:
        idx = sys.argv.index("--pretrained")
        pretrained_ckpt = sys.argv[idx + 1]

    experiment_config_path = os.path.join(
        experiment_directory, FileNames.EXPERIMENT_CONFIG
    )

    results_save_dir = os.path.join(experiment_directory, FolderNames.RESULTS)

    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)

    experiment_config = ExperimentConfig(**load_from_json_file(experiment_config_path))

    run_experiment(
        experiment_config=experiment_config, results_save_dir=results_save_dir,
        resume=resume, pretrained_ckpt=pretrained_ckpt,
    )


if __name__ == "__main__":
    main()
