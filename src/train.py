from src.models import WeatherPrediction
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import wandb
from src.config import ExperimentConfig
from src.constants import FileNames
from src.utils import save_to_json_file
import os


def train_epoch(
    model: WeatherPrediction,
    train_dataloader: DataLoader,
    optimiser: Optimizer,
    loss_fn,
    device,
):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        X, y = batch
        # Removing the batch dimension
        y = y.squeeze(0)

        if len(y.shape) == 3:
            # Removing the extra timestep dimension from y
            y = y.squeeze(-2)
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        outs = model(X=X)
        batch_loss = loss_fn(outs, y)
        batch_loss.backward()
        optimiser.step()
        total_loss += batch_loss.detach().item()

    avg_loss = total_loss / len(train_dataloader)

    return avg_loss


def test(model: WeatherPrediction, test_dataloader: DataLoader, loss_fn, device):
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for batch in test_dataloader:
            X, y = batch
            # Removing the batch dimension
            y = y.squeeze(0)

            if len(y.shape) == 3:
                # Removing the extra timestep dimension from y
                y = y.squeeze(-2)
            X, y = X.to(device), y.to(device)
            outs = model(X=X)
            batch_loss = loss_fn(outs, y)
            total_loss += batch_loss.detach().item()

    avg_loss = total_loss / len(test_dataloader)

    return avg_loss


def train(
    model: WeatherPrediction,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimiser: Optimizer,
    num_epochs: int,
    device: str,
    config: ExperimentConfig,
    results_save_dir: str,
    print_losses: bool = True,
    wandb_log: bool = True,
):
    # Define the loss function
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    test_losses = []

    # Initialize Weights & Biases logging
    if wandb_log:
        wandb.login(key=config.wandb_key)
        wandb.init(
            entity="graphml-group4",
            project="weather-prediction",
            config=dict(config),
            name=config.wandb_name,
        )

    # Early stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Getting initial performance before training
    intial_train_loss = test(
        model=model, test_dataloader=train_dataloader, loss_fn=loss_fn, device=device
    )

    intial_val_loss = test(
        model=model, test_dataloader=val_dataloader, loss_fn=loss_fn, device=device
    )

    intial_test_loss = test(
        model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=device
    )
    
    train_losses.append(intial_train_loss)
    val_losses.append(intial_val_loss)
    test_losses.append(intial_test_loss)
        
    if wandb_log:
        wandb.log(
            {
                "train_loss": intial_train_loss,
                "val_loss": intial_val_loss,
                "test_loss": intial_test_loss,
            }
            )

    # Running training
    for epoch in range(num_epochs):
        epoch_train_loss = train_epoch(
            model=model,
            optimiser=optimiser,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            device=device,
        )

        epoch_val_loss = test(
            model=model, test_dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        epoch_test_loss = test(
            model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        if print_losses:
            print(f"Train loss after epoch {epoch+1}: {epoch_train_loss}")
            print(f"Validation loss after epoch {epoch+1}: {epoch_val_loss}")
            print(f"Test loss after epoch {epoch+1}: {epoch_test_loss}")

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        test_losses.append(epoch_test_loss)

        if wandb_log:
            wandb.log(
                {
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "test_loss": epoch_test_loss,
                }
            )

        epoch_delta = best_val_loss - epoch_val_loss

        # Early stopping logic
        if epoch_delta > config.early_stopping_delta:

            print(
                f"Val loss reduced by {round(best_val_loss - epoch_val_loss, 5)} which is greater than the early stopping delta. Saving best model... \n"
            )

            best_val_loss = epoch_val_loss
            # Save the best model
            torch.save(
                model.state_dict(),
                os.path.join(results_save_dir, FileNames.SAVED_MODEL),
            )

            patience_counter = 0

        else:
            patience_counter += 1
            print(f"Patience counter is now {patience_counter} \n")

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch+1}. Stopping training.")
            break

    # Save final training results
    training_results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
    }
    save_to_json_file(
        data_dict=training_results,
        save_path=os.path.join(results_save_dir, FileNames.SAVED_RESULTS),
    )
    print(f"Training results saved to {results_save_dir}")

    if wandb_log:
        wandb.finish()

    return training_results
