"""Авторегрессионная развёртка в обучении.

Требует torch.

Это сердце обучения и самая трудная для проверки часть: цикл, в котором прогноз
шага становится входом следующего. Ошибка здесь не падает — она портит то, чему
модель учится, и обнаруживается разве что странной кривой обучения.

Проверяем настоящим `train_epoch` с моделью-заглушкой, которая записывает, что
ей подавали на каждом шаге. Так видно не результат, а само поведение развёртки.
"""
import pytest
from conftest import needs_torch

pytestmark = needs_torch

G, C, OBS = 6, 4, 2


@pytest.fixture
def tr():
    import torch
    return torch


def make_spy_model(delta_value=1.0):
    """Модель, прибавляющая постоянную дельту и запоминающая свои входы."""
    import torch
    import torch.nn as nn

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.obs_window = OBS
            self.delta = nn.Parameter(torch.full((C,), float(delta_value)))
            self.seen = []                    # входы по шагам

        def forward(self, X, attention_threshold=0.0, **kw):
            self.seen.append(X.detach().clone())
            N, Gg, _ = X.shape
            return X.view(N, Gg, self.obs_window, C)[:, :, -1, :] * 0 + self.delta

    return Spy()


def make_batch(n_steps, base=0.0):
    """Один батч: вход из OBS кадров и цель из n_steps кадров."""
    import torch
    x = torch.full((1, G, OBS * C), float(base))
    y = torch.zeros(1, G, n_steps * C)
    return x, y


def run_epoch(model, batches, **kw):
    import torch

    from src.train import train_epoch
    opt = torch.optim.SGD(model.parameters(), lr=0.0)   # веса не двигаем
    return train_epoch(model, batches, opt, None, torch.device("cpu"),
                       threshold=0.0, epoch=0, **kw)


# --- как катится состояние ---------------------------------------------------

def test_state_rolls_forward_by_one_frame_per_step(tr):
    """[1, 2, 3, 4] -> [2, 3, 4, прогноз]: старейший кадр уходит, новый встаёт в конец."""
    m = make_spy_model(delta_value=1.0)
    run_epoch(m, [make_batch(3)], current_ar_steps=3, use_residual=True)
    assert len(m.seen) == 3, "шагов развёртки не столько, сколько заказано"

    first = m.seen[0].view(1, G, OBS, C)
    second = m.seen[1].view(1, G, OBS, C)
    # второй вход: последний кадр первого входа сдвинулся на место предпоследнего
    assert tr.allclose(second[:, :, 0, :], first[:, :, 1, :]), "кадр не сдвинулся"
    # а в конец встал прогноз первого шага: вход был нулевым, дельта 1
    assert tr.allclose(second[:, :, 1, :], tr.ones(1, G, C)), "в конец встал не прогноз"


def test_errors_accumulate_along_the_rollout(tr):
    """Модель получает на вход СВОИ ошибки: третий шаг видит уже удвоенную дельту.

    Ради этого развёртка в обучении и нужна — иначе модель никогда не видит
    состояний, которые сама и породила.
    """
    m = make_spy_model(delta_value=1.0)
    run_epoch(m, [make_batch(3)], current_ar_steps=3, use_residual=True)
    last_frames = [s.view(1, G, OBS, C)[:, :, -1, :].mean().item() for s in m.seen]
    assert last_frames == pytest.approx([0.0, 1.0, 2.0], abs=1e-5)


def test_without_residual_the_output_is_the_prediction_itself(tr):
    """Без остаточной формулировки прогноз не прибавляется ко входу, а заменяет его."""
    m = make_spy_model(delta_value=5.0)
    run_epoch(m, [make_batch(2)], current_ar_steps=2, use_residual=False)
    second = m.seen[1].view(1, G, OBS, C)
    assert tr.allclose(second[:, :, -1, :], tr.full((1, G, C), 5.0))


def test_rollout_length_is_capped_by_the_available_targets(tr):
    """Заказали больше шагов, чем есть целей — крутим столько, сколько есть."""
    m = make_spy_model()
    run_epoch(m, [make_batch(2)], current_ar_steps=8, use_residual=True)
    assert len(m.seen) == 2


# --- перенос каналов внутри цикла -------------------------------------------

def test_static_channels_stay_frozen_through_the_rollout(tr):
    """Рельеф не плывёт от шага к шагу, хотя модель прибавляет к нему дельту."""
    m = make_spy_model(delta_value=1.0)
    x = tr.full((1, G, OBS * C), 3.0)
    y = tr.zeros(1, G, 3 * C)
    run_epoch(m, [(x, y)], current_ar_steps=3, use_residual=True,
              static_channels=[0])
    for s in m.seen:
        frame = s.view(1, G, OBS, C)
        assert tr.allclose(frame[:, :, -1, 0], tr.full((1, G), 3.0)), (
            "статический канал уехал")


def test_forcing_channels_come_from_the_target_each_step(tr):
    """Час и день года берутся из цели на каждый шаг, а не предсказываются."""
    m = make_spy_model(delta_value=1.0)
    x = tr.zeros(1, G, OBS * C)
    y = tr.zeros(1, G, 3 * C)
    for step in range(3):
        y[:, :, step * C + 1] = float(step + 10)      # канал 1 — форсинг
    run_epoch(m, [(x, y)], current_ar_steps=3, use_residual=True,
              forcing_channels=[1])
    for step in (1, 2):
        frame = m.seen[step].view(1, G, OBS, C)
        assert tr.allclose(frame[:, :, -1, 1], tr.full((1, G), float(step + 9))), (
            f"на шаге {step} форсинг взят не из цели")


# --- градиент ----------------------------------------------------------------

def test_detached_and_full_rollout_report_the_same_loss(tr):
    """Отцепление шагов меняет градиент, но не значение целевой функции.

    Иначе прогоны с отцеплением и без были бы несравнимы, а вводилось оно ради
    памяти: без него двенадцать шагов на карту не влезают.
    """
    losses = []
    for detach in (False, True):
        m = make_spy_model(delta_value=0.5)
        losses.append(run_epoch(m, [make_batch(4)], current_ar_steps=4,
                                use_residual=True, detach_ar=detach))
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)


def test_gradients_do_not_accumulate_between_batches(tr):
    """Градиент обнуляется на каждом батче.

    Без этого он копился бы от батча к батчу, шаг оптимизатора рос бы и
    обучение разошлось бы — но не сразу, а на середине эпохи.
    """
    import torch

    from src.train import train_epoch
    m = make_spy_model(delta_value=1.0)
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    batches = [make_batch(1) for _ in range(4)]
    train_epoch(m, batches, opt, None, torch.device("cpu"), threshold=0.0, epoch=0,
                current_ar_steps=1, use_residual=True)
    one = m.delta.grad.clone()

    m2 = make_spy_model(delta_value=1.0)
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.0)
    train_epoch(m2, [make_batch(1)], opt2, None, torch.device("cpu"),
                threshold=0.0, epoch=0, current_ar_steps=1, use_residual=True)
    assert torch.allclose(one, m2.delta.grad, rtol=1e-4), (
        "градиент последнего батча зависит от числа предыдущих — не обнуляется")


def test_reported_loss_is_averaged_over_steps(tr):
    """Отчётная потеря — среднее по шагам, а не сумма.

    При сумме длинные развёртки давали бы больший «лосс» просто от длины, и
    кривые обучения разных стадий были бы несравнимы.
    """
    a = run_epoch(make_spy_model(delta_value=1.0), [make_batch(1)],
                  current_ar_steps=1, use_residual=True)
    b = run_epoch(make_spy_model(delta_value=1.0), [make_batch(4)],
                  current_ar_steps=4, use_residual=True)
    # при дельте 1 и нулевой цели ошибка шага k равна k^2; среднее по 4 шагам
    # заметно больше, чем по одному, но НЕ вчетверо больше суммы
    assert b > a
    assert b < 4 * (1 + 4 + 9 + 16) / 4


# --- шум ---------------------------------------------------------------------

def test_noise_is_not_added_after_the_last_step(tr):
    """На последнем шаге шум не подмешивается: его результат никуда не идёт."""
    m = make_spy_model(delta_value=0.0)
    run_epoch(m, [make_batch(1)], current_ar_steps=1, use_residual=True,
              noise_sigma=10.0)
    assert len(m.seen) == 1                     # проверяем, что не упало
    frame = m.seen[0].view(1, G, OBS, C)
    assert tr.allclose(frame, tr.zeros_like(frame)), "вход первого шага зашумлён"


def test_noise_starts_from_the_configured_step(tr):
    """До заданного шага развёртки шум не подмешивается вовсе."""
    m = make_spy_model(delta_value=0.0)
    run_epoch(m, [make_batch(3)], current_ar_steps=3, use_residual=True,
              noise_sigma=5.0, noise_apply_from_ar_step=3)
    second = m.seen[1].view(1, G, OBS, C)[:, :, -1, :]
    assert tr.allclose(second, tr.zeros_like(second), atol=1e-6), (
        "шум подмешан раньше заданного шага")
