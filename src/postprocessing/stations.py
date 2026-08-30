"""Чтение списка станций — в одном месте на все скрипты.

Файл `data/krsk_postproc_stations.json` — это СЛОВАРЬ, где номер станции стоит
ключом, а не полем внутри записи:

    {"287854": {"name": "ABAKAN", "lat": 53.74, "lon": 91.385, "elev": 253.3, ...}}

Отдельные загрузчики в каждом скрипте это уже подводили: 30.08.2026
`add_terrain.py` читал `raw.values()` и падал на `KeyError: 'usaf'`, потому что
номер оставался в выброшенном ключе. Поэтому загрузчик здесь один, и все
читают через него.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_stations(path) -> list[dict]:
    """Список станций, у каждой гарантированно есть usaf, lat, lon.

    Принимает обе формы: словарь с номером в ключе и список записей. Долгота
    приводится к диапазону [-180, 180]: в описаниях станций она встречается и в
    виде 0..360, а листы матрицы высот именуются по отрицательным на западе.
    """
    raw = json.loads(Path(path).read_text())

    if isinstance(raw, dict):
        items = []
        for key, rec in raw.items():
            rec = dict(rec)
            rec.setdefault("usaf", str(key))
            items.append(rec)
    else:
        items = [dict(r) for r in raw]

    out = []
    for i, rec in enumerate(items):
        for field in ("lat", "lon"):
            if field not in rec:
                raise ValueError(
                    f"{path}: у записи {i} нет поля {field!r}. "
                    f"Есть: {sorted(rec)}")
        if "usaf" not in rec:
            raise ValueError(
                f"{path}: у записи {i} нет номера станции. Ожидается либо ключ "
                f"словаря, либо поле 'usaf'. Есть: {sorted(rec)}")
        rec["usaf"] = str(rec["usaf"])
        rec["lat"] = float(rec["lat"])
        lon = float(rec["lon"]) % 360.0
        rec["lon"] = lon - 360.0 if lon > 180.0 else lon
        out.append(rec)
    return out
