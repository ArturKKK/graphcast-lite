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


# --------------------------------------------------------------------------
# Посрочные слагаемые ACC (для доверительных интервалов)
# --------------------------------------------------------------------------

def test_per_sample_acc_terms_sum_to_the_streaming_value():
    """Сумма посрочных слагаемых обязана дать ровно тот же ACC.

    Иначе интервал считался бы по одной величине, а в таблицу шла бы другая —
    расхождение, которое глазами не поймать.
    """
    _acc_terms, = _from_predict("_acc_terms")
    rng = np.random.default_rng(20260920)
    G, C, N = 6, 2, 5
    w = np.abs(rng.normal(size=G)) + 0.1

    store = {f"acc_{k}_region": np.zeros((N, 1, C)) for k in ("num", "ff", "aa")}
    m = StreamingMetrics(C, node_weights=w)
    for i in range(N):
        y = rng.normal(size=(G, C))
        p_ = y + rng.normal(size=(G, C)) * 0.3
        c = rng.normal(size=(G, C)) * 0.5
        m.update(y, p_, clim=c)
        _acc_terms(store, "region", i, 0, y, p_, c, w)

    num = store["acc_num_region"].sum(axis=(0, 1))
    den = np.sqrt(store["acc_ff_region"].sum(axis=(0, 1)) * store["acc_aa_region"].sum(axis=(0, 1)))
    assert (num / den) == pytest.approx(m.acc_per_channel, rel=1e-9)


def test_per_sample_acc_terms_without_weights():
    _acc_terms, = _from_predict("_acc_terms")
    store = {f"acc_{k}_global": np.zeros((1, 1, 1)) for k in ("num", "ff", "aa")}
    y = np.array([[2.0], [4.0]])
    p_ = np.array([[3.0], [5.0]])
    c = np.array([[1.0], [1.0]])
    _acc_terms(store, "global", 0, 0, y, p_, c, None)
    # fa = (2,4), aa = (1,3) → num = 2+12 = 14
    assert store["acc_num_global"][0, 0, 0] == pytest.approx(14.0)
    assert store["acc_ff_global"][0, 0, 0] == pytest.approx(20.0)
    assert store["acc_aa_global"][0, 0, 0] == pytest.approx(10.0)


def test_per_sample_weighted_mse_aggregates_to_the_streaming_value():
    """Взвешенный посрочный MSE обязан свернуться в ту же величину.

    20.09.2026 прогоны шли с --lat-weight, но в npz клался невзвешенный
    посрочный MSE: агрегат из файла давал 74,55 %, а лог того же прогона —
    74,36 %. То есть в таблицу пошло бы одно число, а доверительный интервал
    считался бы для другого, и заметить это можно было только сверкой руками.
    """
    _wmean, = _from_predict("_wmean")
    rng = np.random.default_rng(20260921)
    G, C, N, P = 7, 3, 4, 2
    w = np.abs(rng.normal(size=G)) + 0.5
    m = StreamingMetrics(C, node_weights=w)
    acc = []
    for _ in range(N):
        y = rng.normal(size=(G, C * P))
        p_ = y + rng.normal(size=(G, C * P)) * 0.4
        m.update(y, p_)
        for h in range(P):
            sl = slice(h * C, (h + 1) * C)
            d2 = (p_[:, sl] - y[:, sl]) ** 2
            acc.append(_wmean(d2, w))
    assert np.mean(acc) == pytest.approx(m.mse, rel=1e-9)


def test_unweighted_wmean_is_a_plain_mean():
    _wmean, = _from_predict("_wmean")
    d2 = np.array([[1.0, 3.0], [3.0, 5.0]])
    assert _wmean(d2, None) == pytest.approx([2.0, 4.0])


# --------------------------------------------------------------------------
# Климатология таблицей (WeatherBench 2)
# --------------------------------------------------------------------------

@pytest.fixture
def table_file(tmp_path):
    """Таблица на два узла и два канала из трёх модельных."""
    rng = np.random.default_rng(20260921)
    data = rng.normal(size=(2, 4, 366, 2)).astype(np.float32)   # каналы, час, день, узлы
    f = tmp_path / "tab.npz"
    np.savez(f, clim=data, channels=np.array(["t2m", "msl"]),
             hour=np.array([0, 6, 12, 18]), dayofyear=np.arange(1, 367),
             node_index=np.array([10, 11]), time_start="2010-01-01T00:00:00",
             obs_window=2)
    return f, data


def test_table_climatology_marks_channels_without_data(table_file):
    """Канал без климатологии должен отдавать NaN, а не тихо нули."""
    Climatology, = _from_predict("Climatology")
    f, _ = table_file
    c = Climatology(f, n_channels=3, var_names=["t2m", "10u", "msl"],
                    mean=np.zeros(3), std=np.ones(3))
    got = c.field(t_offset=0, horizon=0)
    assert got.shape == (2, 3)
    assert np.isfinite(got[:, 0]).all(), "t2m должен быть"
    assert np.isnan(got[:, 1]).all(), "у 10u климатологии нет — ожидается NaN"
    assert np.isfinite(got[:, 2]).all(), "msl должен быть"


def test_table_climatology_is_standardised(table_file):
    """Таблица в физических единицах, поля модели нормированы."""
    Climatology, = _from_predict("Climatology")
    f, raw = table_file
    mean = np.array([100.0, 0.0, 7.0])
    std = np.array([2.0, 1.0, 5.0])
    c = Climatology(f, n_channels=3, var_names=["t2m", "10u", "msl"],
                    mean=mean, std=std)
    got = c.field(t_offset=0, horizon=0)
    # шаг 0 при t_offset=0 отвечает 12 UTC 01.01.2010 (окно наблюдений 2)
    want_t2m = (raw[0, 2, 0] - mean[0]) / std[0]
    assert got[:, 0] == pytest.approx(want_t2m, rel=1e-5)


def test_table_climatology_needs_scalers(table_file):
    f, _ = table_file
    Climatology, = _from_predict("Climatology")
    with pytest.raises(SystemExit, match="scalers"):
        Climatology(f, n_channels=3, var_names=["t2m", "10u", "msl"])


def test_metrics_keep_old_measure_where_climatology_missing():
    """Смешивать две меры ACC в одной колонке нельзя.

    Канал без климатологии обязан считаться по-старому (отклонение от среднего
    по области), а не давать NaN и не портить остальные каналы.
    """
    rng = np.random.default_rng(1)
    G, C = 5, 2
    y = rng.normal(size=(G, C))
    p_ = y + rng.normal(size=(G, C)) * 0.2
    cl = np.empty((G, C))
    cl[:, 0] = 0.0
    cl[:, 1] = np.nan
    m = StreamingMetrics(C)
    m.update(y, p_, clim=cl)
    acc = m.acc_per_channel
    assert np.isfinite(acc).all(), "NaN просочился в ACC"
    assert m.acc_num[0] != 0 and m.acc_num[1] == 0
    assert m.sum_acc[0] == 0 and m.sum_acc[1] != 0
    # Канал без климатологии обязан отдать ИМЕННО прежнюю величину, а не ноль.
    # Ноль тоже конечен, и прежняя проверка на конечность его пропускала: на
    # настоящем прогоне это дало ACC 0,21 вместо 0,95.
    assert acc[1] == pytest.approx(m.sum_acc[1] / m.acc_count[1], rel=1e-12)
    assert acc[1] != 0
    assert list(m.acc_true_mask) == [True, False]


def test_aggregate_acc_is_not_diluted_by_channels_without_climatology():
    """Среднее не должно падать оттого, что климатология покрывает не всё."""
    rng = np.random.default_rng(7)
    G, C = 6, 9
    y = rng.normal(size=(G, C))
    p_ = y + rng.normal(size=(G, C)) * 0.05      # почти точный прогноз
    cl = np.full((G, C), np.nan)
    cl[:, :2] = 0.0                              # климатология лишь у двух каналов
    m = StreamingMetrics(C)
    m.update(y, p_, clim=cl)
    assert m.acc > 0.8, f"агрегат просел до {m.acc:.3f} — каналы без климатологии обнулены"


def test_table_climatology_selects_a_subset_of_nodes(table_file):
    """Внутренняя зона — подмножество области, и таблица обязана это уметь."""
    Climatology, = _from_predict("Climatology")
    f, _ = table_file
    c = Climatology(f, n_channels=1, var_names=["t2m"],
                    mean=np.zeros(1), std=np.ones(1))
    full = c.field(0, 0)
    c.select(np.array([11]))            # второй узел таблицы
    one = c.field(0, 0)
    assert one.shape == (1, 1)
    assert one[0, 0] == pytest.approx(full[1, 0], rel=1e-6)


def test_table_climatology_rejects_foreign_nodes(table_file):
    Climatology, = _from_predict("Climatology")
    f, _ = table_file
    c = Climatology(f, n_channels=1, var_names=["t2m"],
                    mean=np.zeros(1), std=np.ones(1))
    with pytest.raises(SystemExit, match="разные наборы точек"):
        c.select(np.array([10, 999]))
