"""Contains the training and testing logic for the Weather Prediction model.

Этот модуль реализует полный цикл обучения/валидации/тестирования модели WeatherPrediction.
(ТВОИ КОММЕНТАРИИ СОХРАНЕНЫ)
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
import numpy as np # Нужно для генерации весов
from datetime import datetime

# --- ЧЕКПОИНТИНГ (для возобновления обучения) ---
def save_checkpoint(path, model, optimiser, epoch, ar_steps, best_val_loss,
                    patience_counter, train_losses, val_losses):
    """Сохраняет полное состояние обучения для возобновления."""
    torch.save({
        'epoch': epoch,
        'ar_steps': ar_steps,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, path)


def load_checkpoint(path, model, optimiser, device):
    """Загружает чекпоинт и возвращает состояние обучения."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimiser.load_state_dict(ckpt['optimizer_state_dict'])
    return {
        'start_epoch': ckpt['epoch'] + 1,  # следующая эпоха
        'ar_steps': ckpt['ar_steps'],
        'best_val_loss': ckpt['best_val_loss'],
        'patience_counter': ckpt['patience_counter'],
        'train_losses': ckpt['train_losses'],
        'val_losses': ckpt['val_losses'],
    }


# --- НОВЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def get_lat_weights(lat_dim, lon_dim, device, flat_lats=None):
    """Создает веса (cos(lat)) чтобы не переобучаться на полюсах.
    
    flat_lats: если задан (np.ndarray shape (N,)), используются реальные широты
               каждого узла вместо linspace. Для мультирезолюционных сеток.
    """
    if flat_lats is not None:
        # Flat grid: веса по реальным координатам каждого узла
        w = torch.cos(torch.deg2rad(torch.from_numpy(flat_lats.copy()).float()))
        w = w / w.mean()
        return w.view(1, -1, 1).to(device)
    
    lats = torch.linspace(-90, 90, lat_dim)
    w = torch.cos(torch.deg2rad(lats))
    w = w / w.mean()
    # Размножаем веса на всю сетку [1, G, 1]
    # Порядок flatten в датасете: сначала Lon, потом Lat (меняется реже) или наоборот?
    # Обычно (Lon, Lat). Расширяем w [Lat] -> [Lon, Lat] -> Flatten
    w_expanded = w.view(1, -1).expand(lon_dim, lat_dim).reshape(-1)
    return w_expanded.view(1, -1, 1).to(device)

def weighted_mse_loss(pred, target, lat_weights=None):
    """Обычный MSE, но с учетом весов широты."""
    diff = (pred - target) ** 2
    if lat_weights is not None:
        diff = diff * lat_weights
    return diff.mean()
# -------------------------------------

def spatial_corr(pred: torch.Tensor, true: torch.Tensor) -> float:
    """ (ТВОЯ ФУНКЦИЯ БЕЗ ИЗМЕНЕНИЙ) """  
    if pred.dim() == 3:      # [B, N, F] -> усредним по батчу  
        pred = pred.mean(dim=0)  
        true = true.mean(dim=0)  

    p = (pred - pred.mean(dim=0, keepdim=True)) / (pred.std(dim=0, keepdim=True) + 1e-8)  
    t = (true - true.mean(dim=0, keepdim=True)) / (true.std(dim=0, keepdim=True) + 1e-8)  
    acc_per_feat = (p * t).mean(dim=0)  
    return acc_per_feat.mean().item()

def update_attention_threshold(epoch, max_epochs=30, start_epoch=5, final_threshold=0.1356):
    """ (ТВОЯ ФУНКЦИЯ БЕЗ ИЗМЕНЕНИЙ) """  
    if epoch < start_epoch: return 0.0  
    if epoch > max_epochs + start_epoch: return final_threshold  
    return min(final_threshold, (epoch - start_epoch) * final_threshold / (max_epochs - start_epoch))

def train_epoch(
    model: WeatherPrediction,
    train_dataloader: DataLoader,
    optimiser: Optimizer,
    loss_fn, # Оставляем для совместимости, но внутри юзаем weighted_mse
    device,
    threshold,
    epoch,
    # Новые аргументы (с дефолтными значениями, чтобы не ломать старое)
    lat_weights=None, 
    current_ar_steps=1 
):
    """Один проход обучения. ТЕПЕРЬ С АВТОРЕГРЕССИЕЙ."""  
    model.train()  
    total_loss = 0  
    # print(threshold) # Можно раскомментировать для отладки

    for i, batch in enumerate(train_dataloader):  
        X, y = batch  
        # Удаляем лишние размерности (если батч 1)
        y = y.squeeze(0) if len(y.shape) == 4 else y 
        X, y = X.to(device), y.to(device)  
        
        optimiser.zero_grad()  

        # --- НАЧАЛО НОВОЙ ЛОГИКИ (AR) ---
        
        # 1. Понимаем размеры.
        # y (цель) содержит ar_steps шагов (pred_window из датасета).
        # Нам нужно разбить y на отдельные шаги.
        N, G, _ = X.shape
        # Определяем число каналов C из размерности входа и obs_window модели
        C = X.shape[-1] // model.obs_window  # число каналов (17 или 15)
        total_target_feats = y.shape[-1]
        total_target_steps = total_target_feats // C  # сколько шагов target в данных
        
        # [N, G, total_target_steps, C]
        y_steps = y.view(N, G, total_target_steps, C)
        
        # 2. Готовим входное состояние
        # X: [N, G, obs*C]. Превращаем в [N, G, obs, C]
        obs = model.obs_window
        curr_state = X.view(N, G, obs, C)
        
        loss_batch = 0
        
        # 3. Крутим цикл (сколько шагов скажет current_ar_steps, но не больше target)
        steps_to_run = min(current_ar_steps, total_target_steps)
        
        for step in range(steps_to_run):
            # Вход в модель (плоский)
            inp = curr_state.view(N, G, -1)
            
            # Прогноз (Модель теперь должна выдавать 1 шаг! C=15)
            out = model(X=inp, attention_threshold=threshold, epoch=epoch, batch_num=i)
            
            # Если батч пропал
            if out.dim() == 2: out = out.unsqueeze(0)
            
            # Цель на этот шаг
            target = y_steps[:, :, step, :]
            
            # Считаем лосс
            loss_batch += weighted_mse_loss(out, target, lat_weights)
            
            # САМОЕ ГЛАВНОЕ: Добавляем наш прогноз в историю для следующего шага
            # [1, 2, 3, 4] -> [2, 3, 4, out]
            out_unsqueezed = out.unsqueeze(2)
            curr_state = torch.cat([curr_state[:, :, 1:, :], out_unsqueezed], dim=2)
            
        # Усредняем лосс и делаем шаг
        loss_batch = loss_batch / steps_to_run
        loss_batch.backward()
        optimiser.step()
        
        total_loss += loss_batch.detach().item()  
        # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    avg_loss = total_loss / len(train_dataloader)  
    return avg_loss

def test(model: WeatherPrediction, test_dataloader: DataLoader, loss_fn, device, lat_weights=None):
    """Оценка модели. Для скорости проверяем только 1-й шаг."""
    model.eval()

    total_loss = 0  
    acc_values = []  

    with torch.no_grad():  
        for batch in test_dataloader:  
            X, y = batch  
            y = y.squeeze(0) if len(y.shape) == 4 else y
            X, y = X.to(device), y.to(device)  
            
            # Берем только 1-й шаг из y для сравнения
            C = X.shape[-1] // model.obs_window if hasattr(model, 'obs_window') else y.shape[-1]
            total_target_steps = y.shape[-1] // C if C > 0 else 1
            if total_target_steps > 1:
                y_step0 = y.view(y.shape[0], y.shape[1], total_target_steps, C)[:, :, 0, :]
            else:
                y_step0 = y

            outs = model(X=X, attention_threshold=0.0) 
            if outs.dim() == 2: outs = outs.unsqueeze(0)
            
            # Используем тот же лосс
            loss = weighted_mse_loss(outs, y_step0, lat_weights)
            
            total_loss += loss.item()  
            acc_values.append(spatial_corr(outs, y_step0))  

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
    # НОВОЕ: Метаданные нужны, чтобы узнать размеры сетки для весов
    dataset_metadata=None, 
    print_losses: bool = True,
    wandb_log: bool = True,
    resume_checkpoint: str = None,  # путь к checkpoint.pth для возобновления
):
    # --- Инициализация весов (Новое) ---
    lat_weights = None
    if config.use_latitude_weighting and dataset_metadata:
        if getattr(dataset_metadata, 'flat_grid', False) and hasattr(dataset_metadata, 'cordinates'):
            # Flat grid: используем реальные широты узлов
            flat_lats = dataset_metadata.cordinates[0]
            lat_weights = get_lat_weights(0, 0, device, flat_lats=flat_lats)
        else:
            lat_weights = get_lat_weights(dataset_metadata.num_latitudes, dataset_metadata.num_longitudes, device)
        print("[Train] Включен Weighted Loss.")
        
    # --- Инициализация Curriculum (Новое) ---
    ar_steps = 1
    max_ar = config.max_ar_steps # 4
    epochs_per_stage = num_epochs // max_ar if max_ar > 0 else num_epochs

    loss_fn = nn.MSELoss()

    train_losses = []  
    val_losses = []  
    test_losses = []  

    if wandb_log:  
        wandb.login(key=config.wandb_key)  
        wandb.init(entity="graphml-group4", project="weather-prediction", config=dict(config), name=config.wandb_name)  

    best_val_loss = float("inf")  
    patience_counter = 0
    start_epoch = 0  # с какой эпохи начинаем

    # --- Возобновление из чекпоинта ---
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt_state = load_checkpoint(resume_checkpoint, model, optimiser, device)
        start_epoch = ckpt_state['start_epoch']
        ar_steps = ckpt_state['ar_steps']
        best_val_loss = ckpt_state['best_val_loss']
        patience_counter = ckpt_state['patience_counter']
        train_losses = ckpt_state['train_losses']
        val_losses = ckpt_state['val_losses']
        print(f"\n>>> ВОЗОБНОВЛЕНИЕ с эпохи {start_epoch + 1}, AR={ar_steps}, "
              f"best_val_loss={best_val_loss:.5f}, patience={patience_counter} <<<\n")

    # --- File logging (можно отключить nohup и просто смотреть файл) ---
    log_path = os.path.join(results_save_dir, "training_log.txt")
    def _log(msg):
        """Пишем строку и в stdout, и в файл."""
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    _log(f"=== Training started: {datetime.now().isoformat()} ===")
    _log(f"epochs={num_epochs}  max_ar={max_ar}  epochs_per_stage={epochs_per_stage}")
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        _log(f">>> Resumed from epoch {start_epoch}, AR={ar_steps}, best_vl={best_val_loss:.5f}")
    _log(f"{'epoch':>5}  {'ar':>2}  {'train_loss':>10}  {'val_loss':>10}  {'val_ACC':>8}  {'best_vl':>10}  {'patience':>8}  timestamp")
    _log("-" * 90)

    if start_epoch == 0:
        intial_val_loss, initial_val_acc = test(model, val_dataloader, loss_fn, device, lat_weights)  
        if print_losses:  
            print(f"[Init] val_loss={intial_val_loss:.5f} val_acc={initial_val_acc:.4f}")
        _log(f"{'init':>5}  {'--':>2}  {'--':>10}  {intial_val_loss:10.5f}  {initial_val_acc:8.4f}  {'--':>10}  {'--':>8}  {datetime.now().strftime('%H:%M:%S')}")

    # Основной цикл обучения  
    for epoch in range(start_epoch, num_epochs):  
        print()
        
        # --- Увеличение сложности ---
        # Вычисляем правильный AR-уровень для текущей эпохи
        # (важно при resume: epoch может быть > 0 с первой итерации)
        correct_ar = min(1 + epoch // epochs_per_stage, max_ar)
        if correct_ar > ar_steps:
            ar_steps = correct_ar
            print(f"\n>>> УРОВЕНЬ СЛОЖНОСТИ ПОВЫШЕН! Теперь обучаем на {ar_steps} шага(ов) вперед. <<<\n")
            
            # ВАЖНО: Сбрасываем счетчик Early Stopping!
            # Даем модели "фору", чтобы привыкнуть к новой сложности.
            patience_counter = 0

        epoch_threshold = update_attention_threshold(epoch)  
        print(f"Epoch {epoch} (AR={ar_steps}) with attention threshold {epoch_threshold}")  

        # Передаем новые параметры
        epoch_train_loss = train_epoch(  
            model, train_dataloader, optimiser, loss_fn, device, 
            epoch_threshold, epoch, 
            lat_weights=lat_weights, current_ar_steps=ar_steps
        )  

        epoch_val_loss, epoch_val_acc = test(  
            model, val_dataloader, loss_fn, device, lat_weights
        )  

        if print_losses:  
            print(f"[Epoch {epoch+1}] train_loss={epoch_train_loss:.5f}  val_loss={epoch_val_loss:.5f}  val_ACC={epoch_val_acc:.4f}")  

        train_losses.append(epoch_train_loss)  
        val_losses.append(epoch_val_loss)  

        if wandb_log:  
            wandb.log({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss, "val_acc": epoch_val_acc, "epoch": epoch + 1, "ar_steps": ar_steps})  

        epoch_delta = best_val_loss - epoch_val_loss  

        if epoch_delta > config.early_stopping_delta:  
            print(f"Val loss reduced by {round(best_val_loss - epoch_val_loss, 5)}. Saving best model... \n")  
            best_val_loss = epoch_val_loss  
            torch.save(model.state_dict(), os.path.join(results_save_dir, FileNames.SAVED_MODEL))  
            patience_counter = 0  
        else:  
            patience_counter += 1  
            print(f"Patience counter is now {patience_counter} \n")  

        # --- Пишем строку в лог-файл ---
        _log(f"{epoch+1:5d}  {ar_steps:2d}  {epoch_train_loss:10.5f}  {epoch_val_loss:10.5f}  {epoch_val_acc:8.4f}  {best_val_loss:10.5f}  {patience_counter:8d}  {datetime.now().strftime('%H:%M:%S')}")

        # --- Сохраняем чекпоинт для возможного возобновления ---
        save_checkpoint(
            path=os.path.join(results_save_dir, FileNames.CHECKPOINT),
            model=model, optimiser=optimiser, epoch=epoch,
            ar_steps=ar_steps, best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            train_losses=train_losses, val_losses=val_losses,
        )

        if patience_counter >= config.early_stopping_patience:  
            print(f"Early stopping.")  
            _log(f">>> Early stopping at epoch {epoch+1}")
            break  

    _log(f"=== Training finished: {datetime.now().isoformat()} ===")
    training_results = {"train_losses": train_losses, "val_losses": val_losses}  
    save_to_json_file(training_results, os.path.join(results_save_dir, FileNames.SAVED_RESULTS))  
    if wandb_log: wandb.finish()  

    return training_results
