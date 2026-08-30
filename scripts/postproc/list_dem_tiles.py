#!/usr/bin/env python3
"""Какие листы матрицы высот нужны для наших станций — и как их скачать.

Запускать НА МАШИНЕ С ИНТЕРНЕТОМ (на виртуалке его нет). Скрипт печатает список
листов и готовые команды загрузки; сами данные потом переносятся на виртуалку.

Формат .hgt выбран потому, что читается одним numpy: сырые двухбайтовые целые с
обратным порядком байт, без заголовков. GeoTIFF потребовал бы rasterio или GDAL,
а ставить их ради статичной таблицы признаков незачем.

Источник — открытый набор высот на AWS, без учётной записи и ключей.

    python3 scripts/postproc/list_dem_tiles.py --stations data/krsk_postproc_stations.json
    python3 scripts/postproc/list_dem_tiles.py --stations ... --script fetch_dem.sh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocessing.terrain import tiles_for_points

BASE = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", required=True, help="json со списком станций")
    ap.add_argument("--margin-deg", type=float, default=0.35,
                    help="запас вокруг станции; 0,35° покрывает круг в 20 км "
                         "с избытком даже на юге области")
    ap.add_argument("--script", default=None, help="куда записать команды загрузки")
    ap.add_argument("--out-dir", default="data/dem", help="куда лягут листы")
    a = ap.parse_args()

    raw = json.loads(Path(a.stations).read_text())
    items = raw if isinstance(raw, list) else list(raw.values())
    lats = np.array([float(s["lat"]) for s in items])
    lons = np.array([float(s["lon"]) % 360.0 for s in items])
    lons = np.where(lons > 180.0, lons - 360.0, lons)

    tiles = tiles_for_points(lats, lons, a.margin_deg)
    print(f"станций: {len(items)}, широты {lats.min():.2f}..{lats.max():.2f}, "
          f"долготы {lons.min():.2f}..{lons.max():.2f}")
    print(f"нужно листов: {len(tiles)} (примерно {len(tiles) * 3} МБ в сжатом виде)")
    print(" ".join(tiles))

    total = len(tiles)
    lines = ["#!/usr/bin/env bash",
             "# Загрузка листов матрицы высот. Запускать там, где есть интернет.",
             "# Потом перенести каталог на виртуалку: rsync -a data/dem/ ...",
             "set -uo pipefail",
             f'OUT="${{1:-{a.out_dir}}}"', 'mkdir -p "$OUT"',
             'ok=0; fail=0; had=0',
             f'echo "листов: {total}, каталог: $OUT"']
    for i, t in enumerate(tiles, 1):
        # Ход загрузки печатается ДО curl и без перевода строки: иначе на
        # несколько минут молчания не отличить работу от зависания. curl -fsS
        # сам не выводит ничего, и это уже один раз сбило с толку.
        #
        # Готовность листа проверяется gzip -t, а не наличием файла: прерванная
        # загрузка оставляет обрезанный архив, и проверка «файл есть» пропустила
        # бы его. Обнаружилось бы это уже на виртуалке, порчей мозаики высот.
        lines.append(
            f'printf "[%s/{total}] {t} " "{i}"; '
            f'if gzip -t "$OUT/{t}.hgt.gz" 2>/dev/null; then '
            f'had=$((had+1)); echo "уже есть"; '
            f'elif curl -fsS --retry 3 --retry-delay 2 -o "$OUT/{t}.hgt.gz" '
            f'{BASE}/{t[:3]}/{t}.hgt.gz && gzip -t "$OUT/{t}.hgt.gz" 2>/dev/null; '
            f'then ok=$((ok+1)); echo "ок"; '
            f'else fail=$((fail+1)); rm -f "$OUT/{t}.hgt.gz"; '
            f'echo "НЕТ (бывает: над морем листов не существует)"; fi')
    lines += ['echo "скачано: $ok, уже было: $had, не найдено: $fail"',
              'echo "перенести на виртуалку:"',
              'echo "  rsync -avz \\"$OUT/\\" root@ВИРТУАЛКА:/data/dem/"',
              '# Отсутствие отдельных листов не беда, если они над водой или вне области.']
    text = "\n".join(lines) + "\n"

    if a.script:
        Path(a.script).write_text(text)
        Path(a.script).chmod(0o755)
        print(f"\nкоманды загрузки записаны: {a.script}")
        print("САМИ ДАННЫЕ ЕЩЁ НЕ СКАЧАНЫ. Загрузка — отдельный шаг:")
        print(f"    bash {a.script} {a.out_dir}")
    else:
        print("\nчтобы получить готовый скрипт загрузки, добавь --script fetch_dem.sh")


if __name__ == "__main__":
    main()
