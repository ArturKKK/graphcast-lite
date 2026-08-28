import torch
import numpy as np
import math
from typing import Optional, Sequence

# ==========================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (МАСКИ)
# ==========================================

def build_feature_mask(all_features: Sequence[str], assimilate_features: Sequence[str], pred_window: int, device: torch.device) -> torch.Tensor:
    """
    Строит маску для выбора переменных (по именам).
    """
    C = len(all_features)
    select = torch.zeros(C, dtype=torch.bool)
    name_to_idx = {n: i for i, n in enumerate(all_features)}
    for name in assimilate_features:
        if name in name_to_idx: select[name_to_idx[name]] = True
    return select.repeat(pred_window).to(device)

def build_feature_mask_from_indices(indices: Sequence[int], num_features: int, pred_window: int, device: torch.device) -> torch.Tensor:
    """
    Строит маску для выбора переменных (по индексам).
    """
    C = int(num_features)
    select = torch.zeros(C, dtype=torch.bool)
    for idx in indices:
        if 0 <= idx < C: select[idx] = True
    return select.repeat(pred_window).to(device)

# ==========================================
# 2. ФУНКЦИИ ДЛЯ ГРАНИЦ (TAPERING)
# ==========================================

def cosine_taper_2d(n_rows: int, n_cols: int, border: int) -> torch.Tensor:
    """Косинусное окно (Ханна): единица в середине, спад до нуля у краёв.

    Возвращает (n_rows, n_cols). Порядок осей важен: маска потом выпрямляется и
    накладывается на узлы сетки, а те выпрямлены как (широта, долгота).
    """
    if border <= 0:
        return torch.ones(n_rows, n_cols)

    def hann(n, b):
        # Спад с двух сторон не должен перекрываться в середине: иначе вторая
        # запись затрёт первую, и окно выйдет несимметричным и бессмысленным.
        b = min(int(b), n // 2)
        w = np.ones(n, dtype=np.float32)
        if b <= 0:
            return w
        t = np.linspace(0, 1, b)
        win = 0.5 * (1 - np.cos(np.pi * t))
        w[:b] = win
        w[-b:] = win[::-1]
        return w

    return torch.from_numpy(np.outer(hann(n_rows, border),
                                     hann(n_cols, border))).float()


def build_boundary_taper_mask(height, width, width_x, width_y):
    """Плоская маска [G] для сшивания границ, в порядке узлов сетки.

    Узлы выпрямлены как (широта, долгота): широта меняется медленно, долгота
    быстро. Поэтому окно строится (высота, ширина) и в таком же порядке
    выпрямляется. До 28.08.2026 оно строилось (ширина, высота) и выпрямлялось
    поперёк: на сетке 4x6 треть узлов получала чужой вес, а на 512x256 маска
    была бы бессмысленной целиком. Путь используется только при сшивании
    границ, в прогонах статьи он не участвовал.
    """
    return cosine_taper_2d(height, width, max(width_x, width_y)).ravel()

# ==========================================
# 3. КЛАСС АССИМИЛЯТОРА (NUDGING)
# ==========================================

class NudgingAssimilator:
    def __init__(self, alpha=0.25, device='cpu', feature_mask_flat=None, **kwargs):
        self.alpha = float(alpha)
        self.device = device
        self.mask_flat = feature_mask_flat
        if self.mask_flat is not None: self.mask_flat = self.mask_flat.to(device)

    def apply(self, forecast, observation):
        """
        Применяет формулу Nudging: x_new = x_old + alpha * (y_obs - x_old)
        forecast: [G, Channels]
        observation: [G, Channels] (может содержать NaN)
        """
        # Несовпадение размерностей. Возвращаем прогноз как есть, чтобы не
        # ронять счёт, но МОЛЧА этого делать нельзя: со стороны такой прогон
        # выглядит как «усвоение применили, а выигрыша нет» — и вывод о
        # бесполезности усвоения был бы ложным. Предупреждаем один раз.
        if forecast.shape != observation.shape:
            if not getattr(self, "_warned_shape", False):
                print(f"[nudging] ВНИМАНИЕ: формы не совпали "
                      f"({tuple(forecast.shape)} и {tuple(observation.shape)}) — "
                      f"усвоение НЕ ПРИМЕНЯЕТСЯ, прогноз возвращён как есть",
                      flush=True)
                self._warned_shape = True
            return forecast

        # Маска валидных данных (где есть наблюдения)
        mask = ~torch.isnan(observation)
        
        # Если задана маска фичей (какие переменные усваивать), накладываем её
        if self.mask_flat is not None:
            # mask_flat обычно [Channels]. broadcast если нужно
            if self.mask_flat.shape[0] == forecast.shape[-1]:
                mask = mask & self.mask_flat.unsqueeze(0)
            elif not getattr(self, "_warned_mask", False):
                # Иначе маска каналов молча не применяется, и усваиваются ВСЕ
                # переменные вместо выбранных — то есть считается не тот опыт.
                print(f"[nudging] ВНИМАНИЕ: маска каналов длины "
                      f"{self.mask_flat.shape[0]} не подходит к {forecast.shape[-1]} "
                      f"каналам — усваиваются ВСЕ переменные", flush=True)
                self._warned_mask = True

        analysis = forecast.clone()
        if mask.any():
            diff = observation[mask] - forecast[mask]
            analysis[mask] = forecast[mask] + self.alpha * diff
            
        return analysis

# ==========================================
# 4. ЛОГИКА РОЛЛАУТА (УМНАЯ)
# ==========================================

def sequential_nudged_rollout(model, x0, y_obs, p, alpha=0.25, k=None, device='cpu'):
    """
    Универсальная функция прогноза с усвоением.
    Сама определяет, является ли модель одношаговой (1-step) или многошаговой (multi-step).
    
    Параметры:
      model: обученная нейросеть
      x0: начальное состояние [Batch, G, T_obs, C]
      y_obs: наблюдения на весь период [Batch, G, P*C]
      p: требуемое количество шагов прогноза
      alpha: сила усвоения
    """
    preds = []
    
    # Определяем размерности из входных данных
    N, G, T_in, C = x0.shape
    
    # y_obs приходит плоским по времени [N, G, Total_Target_Channels]
    # Total_Target_Channels должно быть равно P * C
    total_target_channels = y_obs.shape[-1]
    
    # Проверка на всякий случай
    if total_target_channels != p * C:
        # Если не совпадает, пытаемся угадать P
        # (но лучше доверять переданному p)
        pass

    curr_state = x0.to(device)
    nudger = NudgingAssimilator(alpha=alpha, device=device)
    
    # --- ШАГ 1: Пробуем сделать прогноз ---
    
    # Готовим вход: [Batch, G, T_in*C]
    inp = curr_state.view(N, G, -1)
    
    with torch.no_grad():
        # out: [Batch, G, Model_Out_Channels]
        out = model(inp, attention_threshold=0.0).cpu()
    
    # Если батч пропал (модель сделала squeeze), вернем его
    if out.dim() == 2: out = out.unsqueeze(0)
    
    model_out_channels = out.shape[-1]
    
    # --- ШАГ 2: Определяем тип модели ---
    
    # ВАРИАНТ А: Модель предсказала ВСЁ сразу (4pred модель)
    if model_out_channels == total_target_channels:
        # Мы не можем делать цикл авторегрессии, так как модель уже выдала весь горизонт.
        # Мы можем только применить усвоение к результату (Offline Nudging).
        
        # Применяем Nudging ко всему тензору сразу
        for i in range(N):
            out[i] = nudger.apply(out[i], y_obs[i])
            
        return out # [N, G, P*C]

    # ВАРИАНТ Б: Модель предсказывает 1 шаг (1pred модель)
    elif model_out_channels == C:
        # Нам нужно крутить цикл P раз
        
        # Разворачиваем наблюдения по шагам: [N, G, P, C]
        y_obs_reshaped = y_obs.view(N, G, p, C)
        
        # Обрабатываем первый шаг (который уже предсказан выше)
        obs_step_0 = y_obs_reshaped[:, :, 0, :]
        
        if k is None or 0 < k:
            for i in range(N):
                out[i] = nudger.apply(out[i], obs_step_0[i])
        
        preds.append(out) # Добавляем 1-й шаг
        
        # Крутим цикл для остальных шагов (2..P)
        for step in range(1, p):
            # Сдвигаем окно: [N, G, T_in, C] -> выкидываем старое, добавляем out
            out_dev = out.to(device)
            curr_state = torch.cat([curr_state[:, :, 1:, :], out_dev.unsqueeze(2)], dim=2)
            
            # Новый прогноз
            inp = curr_state.view(N, G, -1)
            with torch.no_grad():
                out = model(inp, attention_threshold=0.0).cpu()
            if out.dim() == 2: out = out.unsqueeze(0)
            
            # Усвоение
            obs_step = y_obs_reshaped[:, :, step, :]
            if k is None or step < k:
                for i in range(N):
                    out[i] = nudger.apply(out[i], obs_step[i])
            
            preds.append(out)
            
        # Собираем: list([N, G, C]) -> [N, G, P*C]
        return torch.stack(preds, dim=2).view(N, G, -1)

    else:
        # Какой-то экзотический случай (например, модель предсказывает 2 шага из 4)
        # Для диплома это вряд ли нужно, пока просто вернем что есть
        return out

def nudge_sequence_offline(y_pred, y_obs, alpha=0.25, k=None):
    """Применяет Nudging к готовому тензору прогноза."""
    mask = ~torch.isnan(y_obs)
    y_nudged = y_pred.clone()
    if mask.any():
        y_nudged[mask] = (1 - alpha) * y_pred[mask] + alpha * y_obs[mask]
    return y_nudged
