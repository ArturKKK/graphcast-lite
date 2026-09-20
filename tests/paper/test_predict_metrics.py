"""Широтное взвешивание и ACC против климатологии в StreamingMetrics.

Две величины, обе обещаны рецензентам и обе легко посчитать неправильно так,
что никто не заметит: числа останутся правдоподобными.

Про ACC. До 20.09.2026 в predict.py величина с подписью ACC считалась как
корреляция после вычитания СРЕДНЕГО ПО ОБЛАСТИ. Это не аномальная корреляция:
сезонный ход общий у прогноза и у истины, и, не убрав его климатологией,
получаешь завышенную оценку. Здесь проверяется, что при поданной климатологии
считается формула WeatherBench 2, а без неё — прежняя величина, чтобы старые
прогоны остались сопоставимыми.
"""
import ast
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_climatology import N_COEF, design_row  # noqa: E402


def _from_predict(*names):
    """Берём определения из predict.py, не импортируя его целиком.

    Верхние импорты predict.py тянут src.main, тот — torch.nn и модель.
    Проверяемые здесь вещи от torch не зависят вовсе, поэтому выкусываем их
    разбором исходника: так проверка идёт и там, где нет видеокарты.
    """
    src = (ROOT / "scripts" / "predict.py").read_text()
    tree = ast.parse(src)
    want = set(names)
    nodes = [n for n in tree.body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in want]
    missing = want - {n.name for n in nodes}
    assert not missing, f"в predict.py нет: {missing}"
    ns = {"np": np, "datetime": datetime, "timedelta": timedelta,
          "design_row": design_row}
    exec(compile(ast.Module(nodes, []), "predict.py", "exec"), ns)
    return [ns[n] for n in names]


StreamingMetrics, Climatology, latitude_weights = _from_predict(
    "StreamingMetrics", "Climatology", "latitude_weights")


def mk(vals):
    return np.asarray(vals, dtype=np.float32)


def test_equal_weights_match_unweighted():
    """Веса из одних единиц обязаны дать ровно то же, что и их отсутствие."""
    y = mk([[1.0], [2.0], [3.0], [4.0]])
    p = mk([[1.5], [2.5], [2.0], [4.5]])
    a = StreamingMetrics(1)
    b = StreamingMetrics(1, node_weights=np.ones(4))
    a.update(y, p)
    b.update(y, p)
    assert a.rmse == pytest.approx(b.rmse, rel=1e-12)
    assert a.rmse_per_channel[0] == pytest.approx(b.rmse_per_channel[0], rel=1e-12)


def test_latitude_weighting_shifts_rmse_as_hand_computed():
    """Вес переносит вклад с узла на узел — считаем руками."""
    y = mk([[0.0], [0.0]])
    p = mk([[1.0], [3.0]])          # ошибки 1 и 3
    w = np.array([3.0, 1.0])    # первый узел втрое тяжелее
    m = StreamingMetrics(1, node_weights=w)
    m.update(y, p)
    # sum w*e^2 = 3*1 + 1*9 = 12; sum w = 4 → mse = 3
    assert m.rmse == pytest.approx(np.sqrt(3.0), rel=1e-6)
    # без весов было бы sqrt((1+9)/2) = sqrt(5)
    plain = StreamingMetrics(1)
    plain.update(y, p)
    assert plain.rmse == pytest.approx(np.sqrt(5.0), rel=1e-6)


def test_weights_must_match_node_count():
    """Несовпадение длины весов с маской узлов — ошибка, а не тихий пересчёт."""
    m = StreamingMetrics(1, node_weights=np.ones(4))
    with pytest.raises(ValueError, match="разошлась"):
        m.update(mk([[1.0], [2.0]]), mk([[1.0], [2.0]]))


def test_acc_against_climatology_is_one_for_perfect_forecast():
    y = mk([[1.0], [5.0], [2.0], [4.0]])
    c = mk([[3.0], [3.0], [3.0], [3.0]])
    m = StreamingMetrics(1)
    m.update(y, y, clim=c)
    assert m.acc_is_true
    assert m.acc_per_channel[0] == pytest.approx(1.0, rel=1e-6)


def test_acc_is_zero_when_forecast_equals_climatology():
    """Прогноз «как обычно» не имеет аномалии — корреляция нулевая."""
    y = mk([[1.0], [5.0], [2.0], [4.0]])
    c = mk([[3.0], [3.0], [3.0], [3.0]])
    m = StreamingMetrics(1)
    m.update(y, c, clim=c)
    assert m.acc_per_channel[0] == pytest.approx(0.0, abs=1e-9)


def test_true_acc_differs_from_spatial_mean_version():
    """Ровно та подмена, из-за которой это и переписывалось.

    Климатология здесь неоднородна по узлам, а прогноз сдвинут относительно
    истины. Версия со средним по области этот сдвиг частично съедает.
    """
    y = mk([[1.0], [2.0], [6.0], [7.0]])
    p = mk([[2.0], [1.0], [7.0], [6.0]])
    c = mk([[1.0], [2.0], [5.0], [6.0]])
    old = StreamingMetrics(1)
    old.update(y, p)
    new = StreamingMetrics(1)
    new.update(y, p, clim=c)
    assert not old.acc_is_true and new.acc_is_true
    assert abs(old.acc_per_channel[0] - new.acc_per_channel[0]) > 0.1


def test_acc_accumulates_as_ratio_of_sums_not_mean_of_ratios():
    """Протокол требует отношение сумм: усреднение корреляций даёт другое число."""
    c = mk([[0.0], [0.0]])
    m = StreamingMetrics(1)
    m.update(mk([[1.0], [-1.0]]), mk([[1.0], [-1.0]]), clim=c)      # вклад крупный
    m.update(mk([[0.01], [-0.01]]), mk([[-0.01], [0.01]]), clim=c)  # мелкий, знак обратный
    got = m.acc_per_channel[0]
    mean_of_ratios = (1.0 + (-1.0)) / 2
    assert got > 0.9, "мелкий срок не должен весить столько же, сколько крупный"
    assert abs(got - mean_of_ratios) > 0.5


def test_weighted_acc_uses_the_same_weights():
    y = mk([[1.0], [-1.0]])
    p = mk([[1.0], [1.0]])
    c = mk([[0.0], [0.0]])
    # Второй узел почти невесом → несовпадение по нему почти не считается.
    m = StreamingMetrics(1, node_weights=np.array([1.0, 1e-6]))
    m.update(y, p, clim=c)
    assert m.acc_per_channel[0] == pytest.approx(1.0, abs=1e-3)


def test_per_channel_isolation():
    """Каналы не должны перемешиваться при развёртке C*P."""
    C = 2
    y = mk([[0.0, 0.0], [0.0, 0.0]])
    p = mk([[1.0, 10.0], [1.0, 10.0]])
    m = StreamingMetrics(C)
    m.update(y, p)
    assert m.rmse_per_channel[0] == pytest.approx(1.0, rel=1e-6)
    assert m.rmse_per_channel[1] == pytest.approx(10.0, rel=1e-6)


# --------------------------------------------------------------------------
# Широтные веса
# --------------------------------------------------------------------------

def test_latitude_weights_have_unit_mean():
    """Нормировка нужна, чтобы взвешенная RMSE осталась в прежнем масштабе."""
    w = latitude_weights(np.linspace(-80, 80, 97))
    assert w.mean() == pytest.approx(1.0, rel=1e-12)


def test_latitude_weights_keep_cosine_ratio():
    """Отношение весов 50° и 60° — то самое 1,29 из «Метрик качества»."""
    w = latitude_weights(np.array([50.0, 60.0]))
    assert w[0] / w[1] == pytest.approx(np.cos(np.deg2rad(50)) / np.cos(np.deg2rad(60)), rel=1e-9)
    assert w[0] / w[1] == pytest.approx(1.29, abs=0.01)


def test_latitude_weights_do_not_vanish_at_pole():
    """cos(90°) = 0 обнулил бы полюс и поделил бы на ноль в ACC."""
    w = latitude_weights(np.array([0.0, 90.0]))
    assert w[1] > 0


# --------------------------------------------------------------------------
# Климатология
# --------------------------------------------------------------------------

@pytest.fixture
def clim_file(tmp_path):
    """Климатология из двух узлов и одного канала с известными гармониками."""
    rng = np.random.default_rng(20260920)
    coef = rng.normal(size=(N_COEF, 2, 1)).astype(np.float32)
    f = tmp_path / "coef.npz"
    np.savez(f, coef=coef, time_start="2010-01-01T00:00:00", obs_window=2, n_feat=1)
    return f, coef


def test_climatology_reproduces_the_harmonic_sum(clim_file):
    f, coef = clim_file
    c = Climatology(f, n_channels=1)
    # Шаг 0 при t_offset=0 отвечает сроку t0 + 6ч*(0+2+0) = 12 UTC 01.01.2010
    got = c.field(t_offset=0, horizon=0)
    want = design_row(datetime(2010, 1, 1, 12)).astype(np.float32) @ coef.reshape(N_COEF, -1)
    assert got.ravel() == pytest.approx(want.ravel(), rel=1e-5)


def test_climatology_respects_obs_window_and_horizon(clim_file):
    """Смещение окна наблюдений обязано совпадать с paper_climatology.py.

    Ошибка на один шаг сдвинула бы климатологию на шесть часов и тихо
    испортила бы весь ACC: суточный ход как раз этого порядка.
    """
    f, _ = clim_file
    c = Climatology(f, n_channels=1)
    a = c.field(t_offset=3, horizon=1)
    b = c.field(t_offset=4, horizon=0)   # тот же срок: 3+2+1 == 4+2+0
    assert a == pytest.approx(b, rel=1e-6)


def test_climatology_stacked_matches_fields(clim_file):
    f, _ = clim_file
    c = Climatology(f, n_channels=1)
    st = c.stacked(t_offset=7, n_horizons=3)
    assert st.shape == (2, 3)
    for h in range(3):
        assert st[:, h:h + 1] == pytest.approx(c.field(7, h), rel=1e-6)


def test_climatology_node_subset(clim_file):
    f, _ = clim_file
    c = Climatology(f, n_channels=1)
    idx = np.array([1])
    assert c.field(0, 0, node_idx=idx) == pytest.approx(c.field(0, 0)[idx], rel=1e-9)


def test_climatology_rejects_too_few_channels(clim_file):
    """Молча обрезать каналы нельзя: ACC посчитался бы не по тем полям."""
    f, _ = clim_file
    with pytest.raises(SystemExit, match="каналов"):
        Climatology(f, n_channels=5)
