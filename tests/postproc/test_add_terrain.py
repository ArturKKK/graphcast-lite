"""Приклейка описателей рельефа к корпусу.

Признаки рельефа статичны: одно значение на станцию, одинаковое во всех её
строках. Ошибки здесь тихие и дорогие. Склейка по неуникальному ключу размножит
строки, и корпус вырастет незаметно; перепутанный порядок станций даст каждой
чужой рельеф, и модель обучится на бессмыслице, не выдав ни одной ошибки.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "postproc" / "add_terrain.py"

pytest.importorskip("pyarrow")


def write_hgt(path, arr):
    np.asarray(arr[::-1], dtype=">i2").tofile(str(path))


def hills(n=241, seed=0):
    rng = np.random.default_rng(seed)
    i = np.arange(n)[:, None] / n
    j = np.arange(n)[None, :] / n
    return 300 + 400 * np.sin(6 * i) * np.cos(5 * j) + 30 * rng.standard_normal((n, n))


@pytest.fixture
def scene(tmp_path):
    """Две станции с листами, одна без — как бывает у края области."""
    dem = tmp_path / "dem"
    dem.mkdir()
    for la in (55, 56):
        for lo in (92, 93):
            write_hgt(dem / f"N{la}E0{lo}.hgt", hills(seed=la * 100 + lo))
    stations = [{"usaf": "290000", "lat": 55.4, "lon": 92.6, "elev": 300.0},
                {"usaf": "290001", "lat": 55.8, "lon": 93.4, "elev": 500.0},
                {"usaf": "290002", "lat": 56.2, "lon": 92.2, "elev": 120.0}]
    (tmp_path / "st.json").write_text(json.dumps(stations))
    rows = [{"station_usaf": s["usaf"], "lead_h": 6 * (k % 8),
             "obs_t2m_K": 273.0 + k * 0.01}
            for s in stations for k in range(120)]
    pd.DataFrame(rows).to_parquet(tmp_path / "corpus.parquet", index=False)
    return tmp_path, dem


def run(tmp_path, dem, extra=()):
    out = tmp_path / "out.parquet"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--corpus", str(tmp_path / "corpus.parquet"),
         "--out", str(out), "--stations", str(tmp_path / "st.json"),
         "--dem-dir", str(dem), "--terrain-json", str(tmp_path / "terr.json"), *extra],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return pd.read_parquet(out), r.stdout


def test_row_count_is_preserved(scene):
    """Склейка не размножает и не теряет строк.

    Самая дорогая из возможных ошибок: merge по ключу с повторами раздул бы
    корпус, а обучение прошло бы как ни в чём не бывало — на выборке, где часть
    станций весит вдесятеро больше остальных.
    """
    tmp, dem = scene
    before = len(pd.read_parquet(tmp / "corpus.parquet"))
    got, _ = run(tmp, dem)
    assert len(got) == before


def test_terrain_is_constant_within_a_station(scene):
    """Рельеф не зависит ни от срока, ни от времени — одно значение на станцию."""
    tmp, dem = scene
    got, _ = run(tmp, dem)
    for col in [c for c in got.columns if c.startswith("terr_")]:
        counts = got.groupby("station_usaf")[col].nunique(dropna=True)
        assert (counts <= 1).all(), f"{col} меняется внутри станции"


def test_stations_get_their_own_terrain(scene):
    """Разным станциям достаётся разный рельеф, а не одна строка на всех.

    Проверка против сдвига порядка: если бы таблица клеилась по позиции, а не по
    номеру станции, значения молча разъехались бы по чужим станциям.
    """
    tmp, dem = scene
    got, _ = run(tmp, dem)
    per = got.groupby("station_usaf")["terr_dem_elev"].first().dropna()
    assert len(per) >= 2
    assert per.nunique() == len(per), "у станций совпал рельеф — похоже на склейку по позиции"


def test_missing_tiles_do_not_abort_the_run(scene):
    """Станция без листов остаётся без рельефа, остальные считаются.

    В синтетике лист N55E091 отсутствует, и станция 290002 у края его требует.
    Падать из-за этого нельзя: на настоящих данных так ведут себя станции у
    границы области, и терять из-за одной весь корпус незачем.
    """
    tmp, dem = scene
    got, out = run(tmp, dem)
    assert "без листов осталось станций: 1" in out
    missing = got[got.station_usaf == "290002"]
    assert missing["terr_slope"].isna().all()
    present = got[got.station_usaf == "290000"]
    assert present["terr_slope"].notna().all()


def test_elevation_mismatch_is_measured_against_the_station_record(scene):
    """Расхождение высот считается от заявленной высоты станции.

    Это диагностика координат: расхождение в сотню метров означает, что станция
    стоит не там, где записано, и тогда ВСЕ её описатели сняты не с того места.
    """
    tmp, dem = scene
    got, _ = run(tmp, dem)
    row = got[got.station_usaf == "290000"].iloc[0]
    assert row["terr_elev_mismatch"] == pytest.approx(row["terr_dem_elev"] - 300.0, abs=1e-3)


def test_radii_choice_changes_the_feature_set(scene):
    """Радиусы задаются в километрах и попадают в имена признаков."""
    tmp, dem = scene
    got, _ = run(tmp, dem, extra=("--radii-km", "2", "10"))
    names = {c for c in got.columns if c.startswith("terr_tpi_")}
    assert names == {"terr_tpi_2km", "terr_tpi_10km"}


def test_runs_against_the_real_stations_file(tmp_path):
    """Скрипт запускается на ШТАТНОМ файле станций, а не на самодельном.

    Без этого теста 30.08.2026 всё выглядело зелёным: фикстура выше выдумывала
    станции со всеми нужными полями, а настоящий файл держит номер станции
    КЛЮЧОМ словаря — и скрипт падал с KeyError: 'usaf' уже на виртуалке, после
    часа подготовки. Проверяем на двух станциях, чьи листы синтезируем; для
    остальных ожидаем честное сообщение о нехватке листов.
    """
    real = ROOT / "data" / "krsk_postproc_stations.json"
    if not real.exists():
        pytest.skip("нет штатного файла станций")

    from src.postprocessing.stations import load_stations
    stations = load_stations(real)
    # берём станцию и синтезируем ровно те листы, которые ей нужны
    from src.postprocessing.terrain import tiles_for_points
    s0 = stations[0]
    need = tiles_for_points([s0["lat"]], [s0["lon"]], 0.35)
    dem = tmp_path / "dem"
    dem.mkdir()
    for i, name in enumerate(need):
        write_hgt(dem / f"{name}.hgt", hills(n=181, seed=i))

    rows = [{"station_usaf": s["usaf"], "lead_h": 6, "obs_t2m_K": 273.0}
            for s in stations]
    pd.DataFrame(rows).to_parquet(tmp_path / "corpus.parquet", index=False)

    out = tmp_path / "out.parquet"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--corpus", str(tmp_path / "corpus.parquet"),
         "--out", str(out), "--stations", str(real), "--dem-dir", str(dem)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr

    got = pd.read_parquet(out)
    assert len(got) == len(stations), "склейка изменила число строк"
    mine = got[got.station_usaf == s0["usaf"]]
    assert len(mine) == 1
    assert mine["terr_slope"].notna().all(), "у станции с листами нет рельефа"


def test_station_numbers_match_the_corpus_column(tmp_path):
    """Номер из файла станций стыкуется с колонкой корпуса как строка.

    Если бы одна сторона держала номер числом, а другая строкой, склейка
    молча дала бы пустой рельеф у всех станций — и это выглядело бы как
    «листов не хватило», уводя разбор совсем не туда.
    """
    real = ROOT / "data" / "krsk_postproc_stations.json"
    if not real.exists():
        pytest.skip("нет штатного файла станций")
    from src.postprocessing.stations import load_stations
    for s in load_stations(real):
        assert isinstance(s["usaf"], str)
