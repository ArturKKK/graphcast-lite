"""Служебные части обучения: сохранение состояния и пространственная корреляция.

Требует torch.

Сохранение состояния — то, чем оплачивается возобновление после обрыва. Ошибка
здесь стоит не неверного числа, а потерянного прогона: обучение продолжится не с
того места, и заметить это можно будет только по странной кривой.
"""
import numpy as np
import pytest

from conftest import needs_torch

pytestmark = needs_torch


@pytest.fixture
def helpers():
    from src.train import load_checkpoint, save_checkpoint, spatial_corr
    return dict(save=save_checkpoint, load=load_checkpoint, corr=spatial_corr)


def tiny_model():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


# --- пространственная корреляция ---------------------------------------------

def test_field_correlates_perfectly_with_itself(helpers):
    import torch
    x = torch.randn(500, 3)
    assert helpers["corr"](x, x) == pytest.approx(1.0, abs=1e-3)


def test_inverted_field_correlates_negatively(helpers):
    import torch
    x = torch.randn(500, 3)
    assert helpers["corr"](x, -x) == pytest.approx(-1.0, abs=1e-3)


def test_correlation_ignores_scale_and_shift(helpers):
    """Корреляция смотрит на рисунок поля, а не на его единицы и уровень.

    Это и делает её полезной рядом с ошибкой: она отвечает на другой вопрос —
    угадана ли структура, а не насколько попали в значение.
    """
    import torch
    x = torch.randn(500, 3)
    base = helpers["corr"](x, x)
    assert helpers["corr"](x, 7.0 * x + 100.0) == pytest.approx(base, abs=1e-3)


def test_unrelated_fields_correlate_near_zero(helpers):
    import torch
    g = torch.Generator().manual_seed(0)
    a = torch.randn(4000, 2, generator=g)
    b = torch.randn(4000, 2, generator=g)
    assert abs(helpers["corr"](a, b)) < 0.05


def test_constant_field_gives_zero_not_nan(helpers):
    """Постоянное поле не имеет рисунка — корреляция нулевая, а не NaN.

    NaN здесь разошёлся бы по всему отчёту об эпохе и сделал бы её нечитаемой.
    """
    import torch
    got = helpers["corr"](torch.ones(100, 2), torch.randn(100, 2))
    assert np.isfinite(got) and abs(got) < 1e-3


def test_excluded_channels_do_not_count(helpers):
    """Статические каналы исключаются: они совпадают точно и завышали бы ответ."""
    import torch
    g = torch.Generator().manual_seed(1)
    pred = torch.randn(400, 3, generator=g)
    true = pred.clone()
    true[:, 0] = torch.randn(400, generator=g)      # первый канал угадан плохо
    with_all = helpers["corr"](pred, true)
    without = helpers["corr"](pred, true, exclude_channels=[0])
    assert without > with_all + 0.2
    assert without == pytest.approx(1.0, abs=1e-3)


def test_batched_input_is_averaged_over_samples(helpers):
    import torch
    g = torch.Generator().manual_seed(2)
    a = torch.randn(3, 200, 2, generator=g)
    batched = helpers["corr"](a, a)
    one = helpers["corr"](a[0], a[0])
    assert batched == pytest.approx(one, abs=1e-4)


# --- сохранение и возобновление ---------------------------------------------

def test_checkpoint_round_trip_restores_everything(helpers, tmp_path):
    import torch
    m1, m2 = tiny_model(), tiny_model()
    o1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
    o2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    # сделаем шаг, чтобы у Adam появились моменты
    m1(torch.randn(5, 4)).sum().backward()
    o1.step()

    p = tmp_path / "ckpt.pth"
    helpers["save"](p, m1, o1, epoch=7, ar_steps=3, best_val_loss=0.125,
                    patience_counter=2, train_losses=[1.0, 0.5],
                    val_losses=[1.1, 0.6])
    state = helpers["load"](p, m2, o2, torch.device("cpu"))

    assert state["start_epoch"] == 8, "возобновление должно идти со СЛЕДУЮЩЕЙ эпохи"
    assert state["ar_steps"] == 3
    assert state["best_val_loss"] == pytest.approx(0.125)
    assert state["patience_counter"] == 2
    assert state["train_losses"] == [1.0, 0.5] and state["val_losses"] == [1.1, 0.6]
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b), "веса восстановились не полностью"


def test_optimiser_moments_are_restored(helpers, tmp_path):
    """Моменты Adam переносятся: без них первые эпохи уходят на их восстановление."""
    import torch
    m1, m2 = tiny_model(), tiny_model()
    o1 = torch.optim.Adam(m1.parameters(), lr=1e-3)
    o2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    for _ in range(3):
        o1.zero_grad(); m1(torch.randn(5, 4)).sum().backward(); o1.step()
    p = tmp_path / "c.pth"
    helpers["save"](p, m1, o1, 1, 1, 0.0, 0, [], [])
    helpers["load"](p, m2, o2, torch.device("cpu"))
    got = [s for s in o2.state.values() if "exp_avg" in s]
    assert got, "моменты Adam не восстановились"
    assert any(s["exp_avg"].abs().sum() > 0 for s in got)


def test_mismatched_optimiser_groups_warn_and_continue(helpers, tmp_path, capsys):
    """Разное число групп параметров — не падение, а предупреждение.

    Так бывает при возобновлении без --pretrained: модель обучалась от чужих
    весов с заморозкой процессора, и групп было две. Раньше это падало с
    невнятным сообщением из недр torch и стоило полутора часов.
    """
    import torch
    m1, m2 = tiny_model(), tiny_model()
    o1 = torch.optim.Adam([{"params": list(m1[0].parameters())},
                           {"params": list(m1[2].parameters()), "lr": 1e-4}], lr=1e-3)
    o2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
    p = tmp_path / "c.pth"
    helpers["save"](p, m1, o1, 4, 2, 0.3, 1, [], [])
    state = helpers["load"](p, m2, o2, torch.device("cpu"))
    out = capsys.readouterr().out
    assert "не подошло" in out and "групп в чекпойнте: 2" in out
    assert state["start_epoch"] == 5, "состояние обучения всё равно должно вернуться"
    for a, b in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(a, b), "веса должны загрузиться даже при несовпадении групп"


# --- перенос каналов между шагами развёртки ---------------------------------

@pytest.fixture
def carry():
    from src.train import carry_forward_channels
    return carry_forward_channels


def test_static_channels_come_from_the_input_frame(carry):
    """Рельеф и маска суши берутся с последнего входа, а не из прогноза.

    Оставь там предсказание сети — и рельеф поплывёт от шага к шагу развёртки.
    """
    import torch
    out = torch.zeros(1, 4, 5)
    prev = torch.arange(20, dtype=torch.float32).reshape(1, 4, 5)
    carry(out, prev, None, static_channels=[1, 3])
    assert torch.allclose(out[:, :, 1], prev[:, :, 1])
    assert torch.allclose(out[:, :, 3], prev[:, :, 3])
    assert torch.allclose(out[:, :, 0], torch.zeros(1, 4)), "тронут лишний канал"


def test_forcing_channels_come_from_the_target(carry):
    """Час и день года известны заранее — берутся из цели, а не из прогноза."""
    import torch
    out = torch.zeros(1, 4, 5)
    tgt = torch.full((1, 4, 5), 9.0)
    carry(out, None, tgt, forcing_channels=[4])
    assert torch.allclose(out[:, :, 4], torch.full((1, 4), 9.0))
    assert torch.allclose(out[:, :, :4], torch.zeros(1, 4, 4))


def test_static_and_forcing_together(carry):
    import torch
    out = torch.zeros(1, 3, 6)
    prev = torch.full((1, 3, 6), 1.0)
    tgt = torch.full((1, 3, 6), 2.0)
    carry(out, prev, tgt, static_channels=[0, 1], forcing_channels=[4, 5])
    assert torch.allclose(out[:, :, :2], torch.ones(1, 3, 2))
    assert torch.allclose(out[:, :, 2:4], torch.zeros(1, 3, 2))
    assert torch.allclose(out[:, :, 4:], torch.full((1, 3, 2), 2.0))


def test_missing_target_leaves_forcing_alone(carry):
    """На последнем шаге цели уже нет — форсинг остаётся предсказанным.

    Это штатный случай: подставлять нечего, а падать нельзя.
    """
    import torch
    out = torch.full((1, 2, 3), 7.0)
    carry(out, torch.zeros(1, 2, 3), None, static_channels=[0], forcing_channels=[2])
    assert out[0, 0, 2] == pytest.approx(7.0), "форсинг зря затёрт"
    assert out[0, 0, 0] == pytest.approx(0.0)


def test_empty_channel_lists_change_nothing(carry):
    import torch
    out = torch.randn(1, 3, 4)
    before = out.clone()
    carry(out, torch.zeros(1, 3, 4), torch.ones(1, 3, 4), None, None)
    assert torch.allclose(out, before)


def test_carry_forward_is_idempotent(carry):
    """Применённый дважды даёт то же самое — значит порядок вызовов безопасен."""
    import torch
    prev, tgt = torch.full((1, 2, 4), 3.0), torch.full((1, 2, 4), 5.0)
    a = carry(torch.zeros(1, 2, 4), prev, tgt, [0], [3])
    b = carry(a.clone(), prev, tgt, [0], [3])
    assert torch.allclose(a, b)
