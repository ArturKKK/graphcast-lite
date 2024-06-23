import sys
import os
from src.constants import FileNames, FolderNames
from src.config import ExperimentConfig
from src.data.MultiEdgeData import create_multi_edge_data
from src.models_geometric import WeatherPrediction
from src.utils import load_from_json_file
from torch_geometric.loader import DataLoader
import torch
import numpy as np
from torch.optim import Adam
from src.train import train
from src.data.dataloader import load_train_and_test_datasets
from data.data_loading import WeatherDataset
import random
import matplotlib.pyplot as plt


def set_random_seeds(seed: int = 42):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def load_model_from_experiment_config(
    experiment_config: ExperimentConfig, device
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
        device=device,
    )

    return model


def plot_results(results, results_filename: str):
    train_losses = results["train_losses"]
    test_losses = results["test_losses"]

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(results_filename)
    plt.close()


def run_experiment(experiment_config: ExperimentConfig, results_save_dir: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_random_seeds(seed=experiment_config.random_seed)

    X_train, y_train, X_test, y_test = load_train_and_test_datasets(
        data_path=experiment_config.data.data_directory,
        data_config=experiment_config.data,
    )

    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=experiment_config.batch_size, shuffle=True
    # )
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=experiment_config.batch_size, shuffle=False
    # )

    model: WeatherPrediction = load_model_from_experiment_config(
        experiment_config=experiment_config, device=device
    )

    model = model.to(device)

    # set up data loaders
    edge_index_encoder, edge_index_processor, edge_index_decoder = model.get_edge_indices()

    train_data_list = create_multi_edge_data(X_train, y_train, edge_index_encoder, edge_index_processor, edge_index_decoder)
    train_dataloader = DataLoader(train_data_list, batch_size=experiment_config.batch_size, shuffle=True)

    test_data_list = create_multi_edge_data(X_test, y_test, edge_index_encoder, edge_index_processor, edge_index_decoder)
    test_dataloader = DataLoader(test_data_list, batch_size=experiment_config.batch_size, shuffle=False)


    optimizer = Adam(params=model.parameters(), lr=experiment_config.learning_rate)

    train_losses, test_losses = train(
        model=model,
        train_datalaoder=train_dataloader,
        test_dataloader=test_dataloader,
        optimiser=optimizer,
        num_epochs=experiment_config.num_epochs,
        device=device,
        config=experiment_config,
        print_losses=True,
        wandb_log=False
    )

    results = {
        "train_losses": train_losses,
        "test_losses": test_losses,
    }

    results_filename = os.path.join(results_save_dir, experiment_config.results_filename if experiment_config.results_filename is not None else "test_loss.png")
    plot_results(results, results_filename)

    return results


def main():
    experiment_directory = sys.argv[1]

    experiment_config_path = os.path.join(
        experiment_directory, FileNames.EXPERIMENT_CONFIG
    )

    results_save_dir = os.path.join(experiment_directory, "" if True else FolderNames.RESULTS)

    experiment_config = ExperimentConfig(**load_from_json_file(experiment_config_path))

    run_experiment(
        experiment_config=experiment_config, results_save_dir=results_save_dir
    )

if __name__ == "__main__":
    main()
