from src.models import WeatherPrediction
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm


def train_epoch(
    model: WeatherPrediction,
    train_dataloader: DataLoader,
    optimiser: Optimizer,
    loss_fn,
    device,
): 
    model.train()
    total_loss = 0
    total_samples = 0
    optimiser.zero_grad()
    for batch in train_dataloader:
        X, y = batch
        X, y = X.to(device), y.to(device)
        outs = model(X=X)
        batch_loss = loss_fn(outs, y)
        batch_loss.backward()
        optimiser.step()
        total_loss += batch_loss.detach().item()
        total_samples += X.shape[0]

    avg_loss = total_loss / total_samples

    return avg_loss


def test(model: WeatherPrediction, test_dataloader: DataLoader, loss_fn, device):
    model.eval()

    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_dataloader:
            X, y = batch
            X, y = X.to(device), y.to(device)
            outs = model(X=X)
            batch_loss = loss_fn(outs, y)
            total_loss += batch_loss.detach().item()
            total_samples += X.shape[0]

    avg_loss = total_loss / total_samples

    return avg_loss


def train(
    model: WeatherPrediction,
    train_datalaoder: DataLoader,
    test_dataloader: DataLoader,
    optimiser: Optimizer,
    num_epochs: int,
    device,
):
    # TODO: Make this configurable if we want to combine two losses later.
    loss_fn = nn.MSELoss()

    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        epoch_train_loss = train_epoch(
            model=model,
            optimiser=optimiser,
            train_dataloader=train_datalaoder,
            loss_fn=loss_fn,
            device=device,
        )
        print(f"Train loss after epoch {epoch+1}: {epoch_train_loss}")

        epoch_test_loss = test(
            model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )
        print(f"Test loss after epoch {epoch+1}: {epoch_test_loss}")

        train_losses.append(train_losses)
        test_losses.append(test_losses)

    return train_losses, test_losses
