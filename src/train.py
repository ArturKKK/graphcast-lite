"""Contains the training and testing logic for the Weather Prediction model.

- Этот модуль реализует полный цикл обучения/валидации/тестирования модели `WeatherPrediction`.
- Состоит из:
  * `spatial_corr` — метрика ACC (пространственная корреляция) между предсказанием и истиной по узлам.
  * `update_attention_threshold` — расписание порога для SparseGAT: постепенно увеличивает threshold, чтобы прореживать рёбра по attention.
  * `train_epoch` — одна эпоха обучения: проход по train-дataloader, MSE лосс, оптимизация.
  * `test` — оценка на val/test: без градиентов, считает MSE и ACC.
  * `train` — оркестратор обучения: W&B логирование (опционально), initial eval, цикл эпох с early stopping и сохранением лучшей модели, запись результатов в JSON.

КОНВЕНЦИИ:
- Данные `X`/`y` ожидаются с батчем (часто `B=1`), поэтому внутри идёт `squeeze`.
- Для `WeatherPrediction` можно прокинуть `epoch`/`batch_num`/`attention_threshold` — это нужно для динамического прореживания графа в `SparseGATConv`.
"""

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


def spatial_corr(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Вычисляет среднюю «пространственную корреляцию» (ACC) между предсказанием и истиной по узлам.

    Параметры
    ---------
    pred, true : torch.Tensor
        Формы либо [num_nodes, num_features], либо [batch, num_nodes, num_features].

    Идея
    ----
    1) Если есть батч, усредняем по батчу (получаем среднее поле).
    2) Для каждой фичи нормируем по полю: вычитаем среднее по узлам и делим на std по узлам.
    3) Умножаем нормированные поля поэлементно и усредняем по узлам — это ACC для каждой фичи.
    4) Возвращаем среднее ACC по фичам (скаляр).
    """
    if pred.dim() == 3:      # [B, N, F] -> усредним по батчу
        pred = pred.mean(dim=0)
        true = true.mean(dim=0)

    # Нормируем каждую фичу по полю (по узлам), добавляем eps, чтобы избежать деления на 0
    p = (pred - pred.mean(dim=0, keepdim=True)) / (pred.std(dim=0, keepdim=True) + 1e-8)
    t = (true - true.mean(dim=0, keepdim=True)) / (true.std(dim=0, keepdim=True) + 1e-8)

    # ACC по каждой фиче, потом среднее по фичам
    acc_per_feat = (p * t).mean(dim=0)
    return acc_per_feat.mean().item()


def update_attention_threshold(epoch, max_epochs=30, start_epoch=5, final_threshold=0.1356):
    """
    Расписание порога attention для SparseGAT.

    - До `start_epoch` порог 0.0 (не режем рёбра — модель «разглядывает» весь граф).
    - Затем линейно растём к `final_threshold` в течение `max_epochs - start_epoch` эпох.
    - После (epoch > max_epochs + start_epoch) фиксируемся на `final_threshold`.

    Это помогает сначала стабилизировать обучение, а потом ускоряться и подавлять шумовые связи.
    """
    if epoch < start_epoch:
        return 0.0
    if epoch > max_epochs + start_epoch:
        return final_threshold
    
    # Линейное возрастание от 0.0 до final_threshold
    return min(final_threshold, (epoch - start_epoch) * final_threshold / (max_epochs - start_epoch))


def train_epoch(
    model: WeatherPrediction,
    train_dataloader: DataLoader,
    optimiser: Optimizer,
    loss_fn,
    device,
    threshold,
    epoch
):
    """Один проход обучения по train-датасету.

    Шаги:
    - Переводим модель в train-режим.
    - Для каждого батча: подготавливаем X, y (удаляем лишние размерности, переносим на device),
      прокидываем `epoch`/`batch_num`/`attention_threshold` в модель (нужно для SparseGAT),
      считаем лосс, делаем backward и step.
    - Возвращаем средний лосс по эпохе.
    """
    model.train()
    total_loss = 0
    print(threshold)

    for i, batch in enumerate(train_dataloader):
        X, y = batch
        # Удаляем batch-измерение (ожидается B=1)
        y = y.squeeze(0)

        if len(y.shape) == 3:
            # Если у y есть временное измерение длины 1 — сожмём и его
            y = y.squeeze(-2)
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()

        # Прокидываем вспомогательные аргументы в модель (используются в SparseGATConv)
        kwargs = {
            # "attention_threshold": threshold,  # порог передаём отдельным аргументом
            "epoch": epoch,
            "batch_num": i,
        }
        outs = model(X=X, attention_threshold=threshold, **kwargs)
        batch_loss = loss_fn(outs, y)
        batch_loss.backward()
        optimiser.step()
        total_loss += batch_loss.detach().item()

    avg_loss = total_loss / len(train_dataloader)

    return avg_loss


def test(model: WeatherPrediction, test_dataloader: DataLoader, loss_fn, device):
    """Оценка модели: без градиентов, считаем средний MSE и ACC по даталоудеру."""
    model.eval()

    total_loss = 0
    acc_values = []

    with torch.no_grad():
        for batch in test_dataloader:
            X, y = batch
            # Удаляем batch-измерение (ожидается B=1)
            y = y.squeeze(0)

            if len(y.shape) == 3:
                # Удаляем лишнюю ось времени, если она единичная
                y = y.squeeze(-2)
            X, y = X.to(device), y.to(device)
            outs = model(X=X, attention_threshold=0.0)  # на валидации/тесте порог = 0.0
            batch_loss = loss_fn(outs, y)
            total_loss += batch_loss.detach().item()
            acc_values.append(spatial_corr(outs, y))

    avg_loss = total_loss / len(test_dataloader)
    avg_acc  = sum(acc_values) / max(1, len(acc_values))

    return avg_loss, avg_acc


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
    wandb_log: bool = True, # Если True — логируем метрики/гиперпараметры в Weights & Biases для удобного трекинга экспериментов.
):
    # Функция потерь: среднеквадратичная ошибка
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []
    test_losses = []

    # Инициализируем Weights & Biases (если включено)
    if wandb_log:
        wandb.login(key=config.wandb_key)
        wandb.init(
            entity="graphml-group4",
            project="weather-prediction",
            config=dict(config),
            name=config.wandb_name,
        )

    # Early stopping переменные
    best_val_loss = float("inf")
    patience_counter = 0
    
    # Начальная оценка до обучения (полезно, чтобы видеть «нулевую» точку на графиках)
    intial_train_loss, initial_train_acc = test(
        model=model, test_dataloader=train_dataloader, loss_fn=loss_fn, device=device
    )

    intial_val_loss, initial_val_acc = test(
        model=model, test_dataloader=val_dataloader, loss_fn=loss_fn, device=device
    )

    intial_test_loss, initial_test_acc = test(
        model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=device
    )
    
    train_losses.append(intial_train_loss)
    val_losses.append(intial_val_loss)
    test_losses.append(intial_test_loss)
        
    if print_losses:
        print(f"[Init] train_loss={intial_train_loss:.5f}  val_loss={intial_val_loss:.5f}  test_loss={intial_test_loss:.5f}")
        print(f"[Init] train_ACC={initial_train_acc:.4f}  val_ACC={initial_val_acc:.4f}  test_ACC={initial_test_acc:.4f}")

    if wandb_log:
        wandb.log(
            {
                "train_loss": intial_train_loss,
                "val_loss": intial_val_loss,
                "test_loss": intial_test_loss,
                "train_acc": initial_train_acc,
                "val_acc": initial_val_acc,
                "test_acc": initial_test_acc,
            }
            )

    # Основной цикл обучения
    for epoch in range(num_epochs):
        print()
        epoch_threshold = update_attention_threshold(epoch)
        print(f"Epoch {epoch} with attention threshold {epoch_threshold}")

        epoch_train_loss = train_epoch(
            model=model,
            optimiser=optimiser,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            device=device,
            threshold=epoch_threshold,
            epoch=epoch,
        )

        epoch_val_loss, epoch_val_acc = test(
            model=model, test_dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        epoch_test_loss, epoch_test_acc = test(
            model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        if print_losses:
            print(f"[Epoch {epoch+1}] train_loss={epoch_train_loss:.5f}  val_loss={epoch_val_loss:.5f}  test_loss={epoch_test_loss:.5f}")
            print(f"[Epoch {epoch+1}] val_ACC={epoch_val_acc:.4f}  test_ACC={epoch_test_acc:.4f}")

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        test_losses.append(epoch_test_loss)

        if wandb_log:
            wandb.log(
                {
                    "train_loss": epoch_train_loss,
                    "val_loss": epoch_val_loss,
                    "test_loss": epoch_test_loss,
                    "val_acc": epoch_val_acc,
                    "test_acc": epoch_test_acc,
                    "epoch": epoch + 1,
                }
            )

        epoch_delta = best_val_loss - epoch_val_loss

        # Логика early stopping: если валидационный лосс улучшился больше, чем на delta — сохраняем модель и сбрасываем patience.
        if epoch_delta > config.early_stopping_delta:

            print(
                f"Val loss reduced by {round(best_val_loss - epoch_val_loss, 5)} which is greater than the early stopping delta. Saving best model... \n"
            )

            best_val_loss = epoch_val_loss
            # Сохраняем лучшую модель
            torch.save(
                model.state_dict(),
                os.path.join(results_save_dir, FileNames.SAVED_MODEL),
            )

            patience_counter = 0

        else:
            patience_counter += 1
            print(f"Patience counter is now {patience_counter} \n")

        # Если терпение закончилось — останавливаем обучение
        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered after epoch {epoch+1}. Stopping training.")
            break

    # Сохранение итоговых кривых лоссов (удобно для последующего анализа/визуализации)
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
