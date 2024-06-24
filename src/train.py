from src.models import WeatherPrediction
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
import wandb
from src.config import ExperimentConfig

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
    train_datalaoder: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimiser: Optimizer,
    num_epochs: int,
    device: str,
    config: ExperimentConfig,
    print_losses: bool = True,
    wandb_log: bool = True,
):
    # TODO: Make this configurable if we want to combine two losses later.
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    test_losses = []
    
    if wandb_log:
        wandb.login(key=config.wandb_key)
        wandb.init(
            entity="graphml-group4",
            project="weather-prediction",
            config=dict(config),
            name=config.wandb_name
            )

    for epoch in range(num_epochs):
        epoch_train_loss = train_epoch(
            model=model,
            optimiser=optimiser,
            train_dataloader=train_datalaoder,
            loss_fn=loss_fn,
            device=device,
        )
        if print_losses:
            print(f"Train loss after epoch {epoch+1}: {epoch_train_loss}")

        epoch_val_loss = test(
            model=model, test_dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )
        if print_losses:
            print(f"Validation loss after epoch {epoch+1}: {epoch_val_loss}")
        

        epoch_test_loss = test(
            model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )
        if print_losses:
            print(f"Test loss after epoch {epoch+1}: {epoch_test_loss}")
            print()
             
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        test_losses.append(epoch_test_loss)
        if wandb_log:
            wandb.log({"train_loss": epoch_train_loss, "test_loss": epoch_test_loss, "val_loss": epoch_val_loss})

    if wandb_log:
        wandb.finish()
    return train_losses, test_losses
