import sys
import os
from src.constants import FileNames, FolderNames
from src.config import ExperimentConfig, DataConfig
from src.utils import load_from_json_file
from torch.utils.data import DataLoader
from src.models import WeatherPrediction
import numpy as np
from torch.optim import Adam
from src.train import train
from src.data.dataloader import load_train_and_test_datasets


def load_model_from_experiment_config(
    experiment_config: ExperimentConfig,
) -> WeatherPrediction:
    lats = np.linspace(
        start=-90,
        stop=90,
        num=experiment_config.data.num_latitudes,
        endpoint=True,
    )
    longs = np.linspace(
        start=0,
        stop=360,
        num=experiment_config.data.num_longitudes,
        endpoint=False,
    )

    model = WeatherPrediction(
        cordinates=(lats, longs),
        graph_config=experiment_config.graph,
        pipeline_config=experiment_config.pipeline,
        data_config=experiment_config.data,
    )

    return model


def run_experiment(experiment_config: ExperimentConfig, results_save_dir: str):

    train_dataset, test_dataset = load_train_and_test_datasets(
        data_path=experiment_config.data.data_directory
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=experiment_config.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=experiment_config.batch_size, shuffle=False
    )

    model: WeatherPrediction = load_model_from_experiment_config(
        experiment_config=experiment_config
    )

    optimizer = Adam(params=model.parameters(), lr=experiment_config.learning_rate)

    train_losses, test_losses = train(
        model=model,
        train_datalaoder=train_dataloader,
        test_dataloader=test_dataloader,
        optimiser=optimizer,
        num_epochs=experiment_config.num_epochs,
    )


if __name__ == "__main__":
    experiment_directory = sys.argv[1]

    experiment_config_path = os.path.join(
        experiment_directory, FileNames.EXPERIMENT_CONFIG
    )

    results_save_dir = os.path.join(experiment_directory, FolderNames.RESULTS)

    experiment_config = ExperimentConfig(**load_from_json_file(experiment_config_path))

    run_experiment(
        experiment_config=experiment_config, results_save_dir=results_save_dir
    )
