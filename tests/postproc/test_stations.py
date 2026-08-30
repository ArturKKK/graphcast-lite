"""Чтение списка станций — общий загрузчик для всех скриптов.

Тесты нарочно ходят в НАСТОЯЩИЙ `data/krsk_postproc_stations.json`, а не в
выдуманный. Именно на этом всё и сломалось 30.08.2026: `add_terrain.py` имел
свой загрузчик, тесты кормили его самодельным файлом со всеми нужными полями,
проверки были зелёные — а на настоящих данных он упал с `KeyError: 'usaf'`,
потому что там номер станции стоит КЛЮЧОМ словаря, а не полем записи.

Правило отсюда: если у скрипта есть штатный входной файл, лежащий в
репозитории, хотя бы один тест обязан читать именно его.
"""
import json

import pytest

from src.postprocessing.stations import load_stations

REAL = "data/krsk_postproc_stations.json"


def test_real_file_parses():
    """Штатный файл читается, и станций ровно столько, сколько в корпусе."""
    st = load_stations(REAL)
    assert len(st) == 71


def test_every_real_station_has_what_the_scripts_need():
    """У каждой станции есть номер, координаты и высота.

    Скрипты берут ровно эти поля: номер — чтобы приклеиться к корпусу,
    координаты — чтобы вырезать окрестность из матрицы высот, высоту — чтобы
    сравнить с высотой из матрицы и поймать неточные координаты.
    """
    for s in load_stations(REAL):
        assert s["usaf"] and str(s["usaf"]).isdigit(), s
        assert -90 <= s["lat"] <= 90, s
        assert -180 <= s["lon"] <= 180, s
        assert "elev" in s, s


def test_station_numbers_are_unique():
    """Номера уникальны — иначе склейка с корпусом размножит строки."""
    st = load_stations(REAL)
    nums = [s["usaf"] for s in st]
    assert len(set(nums)) == len(nums)


def test_stations_lie_inside_the_region():
    """Все станции внутри области интереса 50-60 с.ш., 83-98 в.д."""
    for s in load_stations(REAL):
        assert 49.0 <= s["lat"] <= 61.0, s
        assert 82.0 <= s["lon"] <= 99.0, s


def test_number_comes_from_the_dictionary_key(tmp_path):
    """Номер станции стоит ключом словаря — ровно тот случай, что нас подвёл."""
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"287854": {"name": "ABAKAN", "lat": 53.7,
                                        "lon": 91.4, "elev": 253.3}}))
    got = load_stations(p)
    assert got[0]["usaf"] == "287854"
    assert got[0]["name"] == "ABAKAN"


def test_list_form_also_works(tmp_path):
    """Список записей со своим полем usaf читается так же."""
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"usaf": 290000, "lat": 55.0, "lon": 92.0}]))
    got = load_stations(p)
    assert got[0]["usaf"] == "290000", "номер должен приводиться к строке"


def test_longitude_is_brought_into_minus180_180(tmp_path):
    """Долгота 0..360 приводится к -180..180.

    Листы матрицы высот именуются по отрицательной долготе на западе, и без
    приведения станция западнее нуля искала бы лист с несуществующим именем.
    """
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"usaf": "1", "lat": 55.0, "lon": 355.0},
                             {"usaf": "2", "lat": 55.0, "lon": 92.0}]))
    got = load_stations(p)
    assert got[0]["lon"] == pytest.approx(-5.0)
    assert got[1]["lon"] == pytest.approx(92.0)


def test_missing_coordinates_say_what_is_wrong(tmp_path):
    """Нет координат — внятный отказ с перечнем того, что есть.

    Молчаливый пропуск такой записи означал бы станцию без рельефа, и заметить
    это можно было бы только по доле пропусков в самом конце.
    """
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"usaf": "1", "lat": 55.0, "name": "БЕЗ ДОЛГОТЫ"}]))
    with pytest.raises(ValueError, match="lon"):
        load_stations(p)


def test_missing_number_says_what_is_wrong(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([{"lat": 55.0, "lon": 92.0}]))
    with pytest.raises(ValueError, match="номера станции"):
        load_stations(p)
