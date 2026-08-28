"""Геометрия узлов: соседи станции и проверка порядка узлов.

Проверка порядка — единственное, что стоит между разъехавшимися координатами и
двумя с половиной часами счёта впустую. Её первая версия проверяла лишь диапазон
значений и перемешанный порядок пропускала: медиана 261 К, размах 140 К — всё
«в норме». Поэтому здесь проверяется именно то, что она ловит порчу.
"""
import numpy as np
import pytest

from src.postprocessing.geometry import (COHERENCE_MIN_REGIONAL,
                                         COHERENCE_MIN_WHOLE,
                                         field_coherence, neighbour_indices,
                                         snap_miss)


def make_grid(lat0=50.0, lat1=60.0, lon0=83.0, lon1=98.0, step=0.25):
    """Сетка вроде нашей вставки: 0,25° над Красноярским краем."""
    lats = np.arange(lat0, lat1 + 1e-9, step)
    lons = np.arange(lon0, lon1 + 1e-9, step)
    LO, LA = np.meshgrid(lons, lats)
    return LA.ravel(), LO.ravel()


def make_mixed_grid():
    """Крупная сетка с маленькой вставкой — как у нас: 0,703° и 0,25° внутри.

    Вставка берётся не меньше сотни узлов: у проверки есть защитный порог, ниже
    которого она возвращает NaN, потому что на десятке узлов связь ничего не
    значит. В настоящей сетке вставка — 2501 узел из 133 279.
    """
    lats_g, lons_g = make_grid(30, 75, 60, 120, step=0.703)
    lats_r, lons_r = make_grid(55, 57, 92, 95, step=0.25)
    lats = np.concatenate([lats_g, lats_r])
    lons = np.concatenate([lons_g, lons_r])
    is_reg = np.zeros(len(lats), dtype=bool)
    is_reg[len(lats_g):] = True
    return lats, lons, is_reg


def smooth_field(lats, lons):
    """Гладкое поле вроде приземной температуры."""
    return (273.15 + 20 * np.cos(np.radians(lats * 4))
            + 5 * np.sin(np.radians(lons * 3)))


# --- соседи станции ---------------------------------------------------------

def test_neighbours_are_the_nearest_ones():
    lats, lons = make_grid()
    lat, lon = 55.06, 90.06
    idx = neighbour_indices(lats, lons, lat, lon, 6)
    assert len(idx) == 6
    # честное расстояние в километрах
    dy = (lats - lat) * 111.0
    dx = (lons - lon) * 111.0 * np.cos(np.radians(lat))
    d = np.sqrt(dx ** 2 + dy ** 2)
    assert set(idx) == set(np.argsort(d)[:6].tolist())


def test_longitude_is_squeezed_by_latitude():
    """Без поправки в соседи попадает полоса, вытянутая по долготе."""
    lats, lons = make_grid()
    lat, lon = 55.06, 90.06
    dy = (lats - lat) * 111.0
    dx = (lons - lon) * 111.0 * np.cos(np.radians(lat))
    d = np.sqrt(dx ** 2 + dy ** 2)

    good = neighbour_indices(lats, lons, lat, lon, 6)
    naive = np.argsort((lats - lat) ** 2 + (lons - lon) ** 2)[:6]
    assert d[good].max() < d[naive].max(), "поправка на широту не помогла"


def test_zero_neighbours_means_no_neighbours():
    lats, lons = make_grid()
    assert neighbour_indices(lats, lons, 55.0, 90.0, 0) == []


# --- проверка порядка узлов -------------------------------------------------

def test_real_field_passes():
    lats, lons = make_grid()
    assert field_coherence(lats, lons, smooth_field(lats, lons)) > COHERENCE_MIN_WHOLE


def test_scrambled_field_is_caught():
    lats, lons = make_grid()
    f = smooth_field(lats, lons)
    rng = np.random.default_rng(0)
    assert field_coherence(lats, lons, rng.permutation(f)) < COHERENCE_MIN_WHOLE


def test_corrupted_insert_is_invisible_in_the_whole_grid():
    """Главный довод в пользу отдельной проверки вставки.

    Вставка — 1,9 % узлов. Перемешай её целиком, и связь по всей сетке останется
    почти единицей: общая проверка такую порчу пропустит, а прогноз в области,
    ради которой всё и делается, будет мусором.
    """
    lats, lons, is_reg = make_mixed_grid()
    assert is_reg.sum() >= 100, "вставка должна быть выше защитного порога"
    assert is_reg.mean() < 0.05, "вставка в тесте должна быть малой долей узлов"

    f = smooth_field(lats, lons)
    rng = np.random.default_rng(1)
    f[is_reg] = rng.permutation(f[is_reg])

    whole = field_coherence(lats, lons, f)
    reg = field_coherence(lats, lons, f, mask=is_reg)
    assert whole > COHERENCE_MIN_WHOLE, "общая связь должна остаться высокой"
    assert reg < COHERENCE_MIN_REGIONAL, "порча вставки не поймана"


def test_scattered_foreign_values_in_the_insert_are_caught():
    """Во вставку попали значения из случайных узлов сетки.

    Так выглядит настоящая путаница индексов: значения берутся не подряд, а
    вразнобой, и гладкость поля рушится.
    """
    lats, lons, is_reg = make_mixed_grid()
    f = smooth_field(lats, lons)
    rng = np.random.default_rng(7)
    f[is_reg] = f[rng.integers(0, (~is_reg).sum(), is_reg.sum())]
    assert field_coherence(lats, lons, f, mask=is_reg) < COHERENCE_MIN_REGIONAL


def test_smooth_but_wrong_values_can_slip_through():
    """Чего проверка НЕ ловит — и почему с этим можно жить.

    Если во вставку легла гладкая полоса значений из другого места сетки, поле
    остаётся гладким, и связь держится около 0,8 — выше порога. Проверка судит
    о гладкости, а не о правильности значений.

    Мириться с этим можно по двум причинам. Во-первых, так не ломается ни один
    настоящий путь: значения раскладываются по узлам двумя присваиваниями по
    маске, и сбой там даёт разнобой, а не аккуратный сдвиг гладкой полосы.
    Во-вторых, порядок узлов сети всё равно безразличен — обучаемых весов,
    привязанных к номеру узла, у неё нет, а рёбра строятся по координатам.
    Проверка нужна против рассогласования координат и данных, и его она ловит.

    Тест стоит здесь, чтобы граница применимости была записана, а не
    подразумевалась.
    """
    lats, lons, is_reg = make_mixed_grid()
    f = smooth_field(lats, lons)
    f[is_reg] = f[:is_reg.sum()]              # гладкая полоса из другого места
    assert field_coherence(lats, lons, f, mask=is_reg) > COHERENCE_MIN_REGIONAL


def test_constant_field_returns_nan_not_a_false_pass():
    """Постоянное поле — не повод объявлять порядок верным."""
    lats, lons = make_grid()
    assert np.isnan(field_coherence(lats, lons, np.full(len(lats), 273.15)))


def test_normalisation_does_not_change_the_verdict():
    """Связь считается по нормированному кадру, и это законно.

    Нормировка поканальная и линейная, а коэффициент связи к линейному
    преобразованию нечувствителен.
    """
    lats, lons = make_grid()
    f = smooth_field(lats, lons)
    a = field_coherence(lats, lons, f)
    b = field_coherence(lats, lons, (f - 278.9) / 21.1)
    assert a == pytest.approx(b, abs=1e-9)


def test_too_few_nodes_gives_nan_not_a_verdict():
    """На горстке узлов связь не считается — возвращается NaN.

    Это защита от ложного «всё хорошо»: на двух десятках точек коэффициент
    связи скачет как угодно, и решать по нему нельзя.
    """
    lats, lons = make_grid(55, 56, 92, 93, step=0.25)
    assert len(lats) < 100
    assert np.isnan(field_coherence(lats, lons, smooth_field(lats, lons)))


# --- привязка узлов к ячейкам сетки -----------------------------------------

def test_exact_match_gives_no_miss():
    grid = np.arange(0.0, 360.0, 0.703125)
    assert snap_miss(grid, grid[[0, 5, 100, 511]]) == pytest.approx(0.0)


def test_shifted_coordinates_are_caught():
    """Сдвинутые координаты дают заметный промах, а не молчаливую подмену.

    Поиск ближайшей ячейки всегда что-нибудь находит. Без замера промаха кадр
    собрался бы из соседних ячеек, и понять это по метрикам было бы нельзя.
    """
    grid = np.arange(0.0, 360.0, 0.703125)
    assert snap_miss(grid, grid[[10, 20]] + 0.3) == pytest.approx(0.3, abs=1e-9)


def test_meridian_wrap_is_a_real_miss():
    """Круг не замыкается — и это ровно тот случай, ради которого замер нужен.

    Точка на 359,99° при сетке, кончающейся на 359,3°, географически рядом с
    нулевой ячейкой, но поиск по модулю разности выберет 359,3° и промахнётся
    на 0,69°. Проверка это увидит.
    """
    grid = np.arange(0.0, 360.0, 0.703125)
    miss = snap_miss(grid, np.array([359.99]))
    assert miss > 0.5


def test_empty_input_is_not_an_error():
    grid = np.arange(0.0, 10.0, 1.0)
    assert snap_miss(grid, np.array([])) == 0.0
    assert snap_miss(np.array([]), np.array([1.0])) == 0.0
