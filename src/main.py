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
from src.data.data_configs import DatasetMetadata
import random

CURRENT_WORKING_DIR = os.path.dirname(os.path.abspath(__file__))


def set_random_seeds(seed: int = 42):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def load_model_from_experiment_config(
    experiment_config: ExperimentConfig, device, dataset_metadata: DatasetMetadata
) -> WeatherPrediction:

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
    )

    return model


def run_experiment(experiment_config: ExperimentConfig, results_save_dir: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_random_seeds(seed=experiment_config.random_seed)

    train_dataset, val_dataset, test_dataset, dataset_metadata = (
        load_train_and_test_datasets(
            data_path=os.path.join(
                "data", "datasets", experiment_config.data.dataset_name
            ),
            data_config=experiment_config.data,
        )
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=experiment_config.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=experiment_config.batch_size, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=experiment_config.batch_size, shuffle=False
    )

    model: WeatherPrediction = load_model_from_experiment_config(
        experiment_config=experiment_config,
        device=device,
        dataset_metadata=dataset_metadata,
    )

    model = model.to(device)

    optimizer = Adam(params=model.parameters(), lr=experiment_config.learning_rate)

    train_losses, val_losses, test_losses = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimiser=optimizer,
        num_epochs=experiment_config.num_epochs,
        device=device,
        config=experiment_config,
        results_save_dir=results_save_dir,
        print_losses=True,
        wandb_log=experiment_config.wandb_log,
    )

    results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
    }

    return results


def main():
    experiment_directory = sys.argv[1]

    experiment_config_path = os.path.join(
        experiment_directory, FileNames.EXPERIMENT_CONFIG
    )

    results_save_dir = os.path.join(experiment_directory, FolderNames.RESULTS)

    if not os.path.exists(results_save_dir):
        os.makedirs(results_save_dir)

    experiment_config = ExperimentConfig(**load_from_json_file(experiment_config_path))

    run_experiment(
        experiment_config=experiment_config, results_save_dir=results_save_dir
    )


if __name__ == "__main__":
    main()
