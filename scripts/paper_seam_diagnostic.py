#!/usr/bin/env python3
"""S1 — диагностика стыка глобальной и региональной частей графа.

Проверяет главную недоказанную заявку статьи: что жёсткая склейка узлов не
требует отдельной процедуры сшивания, а переходный слой модель формирует сама
в ходе обмена сообщениями. Если это так, ошибка не должна расти по мере
приближения к границе вставки.

Что считает:
  * профиль RMSE приземной температуры по расстоянию до границы вставки —
    отдельно внутри области (региональные узлы 0.25°) и снаружи (глобальные 0.7°);
  * ту же величину по всем горизонтам, чтобы увидеть, накапливается ли эффект
    стыка в авторегрессии;
  * срез поля через границу для карты (сохраняется в .npz).

Отрицательный результат тоже публикуем: всплеск у границы означает, что
формулировку надо смягчить до «контролируемый переходный слой».

Запуск на VM (там лежат предсказания):
    python3 scripts/paper_seam_diagnostic.py \
        --predictions /data/paper_heavy/seam_flagship_preds.pt \
        --data-dir /data/datasets/multires_krsk_19f_merge \
        --out docs/paper/runs/vm4_seam
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

# Границы области интереса (те же, что при построении merge-сетки)
ROI_LAT = (50.0, 60.0)
ROI_LON = (83.0, 98.0)
KM_PER_DEG = 111.19


def dist_to_roi_edge_km(lat, lon):
    """Расстояние до ближайшей границы прямоугольника ROI, км.

    Внутри области — положительное, снаружи — отрицательное. По долготе
    расстояние сжимается косинусом широты, иначе на 60° с. ш. градус долготы
    засчитался бы вдвое длиннее, чем он есть.
    """
    coslat = np.cos(np.radians(lat))
    d_south = (lat - ROI_LAT[0]) * KM_PER_DEG
    d_north = (ROI_LAT[1] - lat) * KM_PER_DEG
    d_west = (lon - ROI_LON[0]) * KM_PER_DEG * coslat
    d_east = (ROI_LON[1] - lon) * KM_PER_DEG * coslat
    inside = (d_south >= 0) & (d_north >= 0) & (d_west >= 0) & (d_east >= 0)
    dmin = np.minimum(np.minimum(d_south, d_north), np.minimum(d_west, d_east))
    # снаружи: расстояние до прямоугольника (по превышению вдоль каждой оси)
    ox = np.maximum(np.maximum(-d_west, -d_east), 0.0)
    oy = np.maximum(np.maximum(-d_south, -d_north), 0.0)
    outside_d = -np.sqrt(ox ** 2 + oy ** 2)
    return np.where(inside, dmin, outside_d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="docs/paper/runs/vm4_seam")
    ap.add_argument("--var", default="t2m")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(a.data_dir)

    # ── координаты узлов и маска вставки ──────────────────────────────
    co = np.load(data_dir / "coords.npz")
    lat, lon = co["latitude"].astype(np.float64), co["longitude"].astype(np.float64)
    is_reg = co["is_regional"].astype(bool)
    print(f"Узлов: {len(lat)} (региональных {is_reg.sum()}, глобальных {(~is_reg).sum()})")

    variables = json.loads((data_dir / "variables.json").read_text())
    ci = variables.index(a.var)
    std = np.load(data_dir / "scalers.npz")["std"][ci]
    print(f"Канал {a.var}: индекс {ci}, разброс {std:.4f}")

    # ── предсказания ──────────────────────────────────────────────────
    print(f"Читаю {a.predictions} …")
    sav = torch.load(a.predictions, map_location="cpu")
    C, P = int(sav["n_features"]), int(sav["ar_steps"])
    cols = [p * C + ci for p in range(P)]
    # берём только нужный канал и сразу освобождаем тяжёлые тензоры
    pred = sav["predictions"][:, :, cols].numpy().astype(np.float64)
    gt = sav["ground_truth"][:, :, cols].numpy().astype(np.float64)
    del sav
    N = pred.shape[0]
    print(f"Сроков: {N}, горизонтов: {P}, узлов: {pred.shape[1]}")

    err = (pred - gt) * std          # физические единицы, (N, G, P)
    d = dist_to_roi_edge_km(lat, lon)

    # ── профиль по расстоянию до границы ──────────────────────────────
    bins_in = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 1000)]
    bins_out = [(-100, -1e-9), (-300, -100), (-1000, -300)]

    lines = ["# Диагностика стыка: RMSE по расстоянию до границы вставки", ""]
    lines.append(f"Переменная {a.var}, физические единицы. Сроков: {N}. "
                 f"Положительное расстояние — внутри вставки (0.25°), "
                 f"отрицательное — снаружи, на глобальной сетке (0.7°).")
    lines.append("")
    lines.append("| Зона | Расстояние до границы, км | Узлов | " +
                 " | ".join(f"+{6*(p+1)} ч" for p in range(P)) + " |")
    lines.append("|---|---|---:|" + "---:|" * P)

    rows = []
    for lo, hi in bins_in:
        m = is_reg & (d >= lo) & (d < hi)
        if m.sum() == 0:
            continue
        r = [np.sqrt((err[:, m, p] ** 2).mean()) for p in range(P)]
        rows.append((f"вставка 0.25°", f"{lo}–{hi}", int(m.sum()), r))
    for lo, hi in bins_out:
        m = (~is_reg) & (d >= lo) & (d < hi)
        if m.sum() == 0:
            continue
        r = [np.sqrt((err[:, m, p] ** 2).mean()) for p in range(P)]
        rows.append((f"глобальная 0.7°", f"{int(lo)}–{int(hi)}", int(m.sum()), r))

    for zone, rng, n, r in rows:
        lines.append(f"| {zone} | {rng} | {n} | " +
                     " | ".join(f"{v:.2f}" for v in r) + " |")

    # ── вывод ─────────────────────────────────────────────────────────
    inner = [row for row in rows if row[0].startswith("вставка")]
    if len(inner) >= 2:
        near, far = inner[0][3][-1], inner[-1][3][-1]
        delta = (near - far) / far * 100
        lines += ["", f"**Полоса у границы против середины области, срок +{6*P} ч: "
                      f"{near:.2f} против {far:.2f} — разница {delta:+.1f} %.**", ""]
        if abs(delta) < 5:
            lines.append("Разница в пределах нескольких процентов: систематического "
                         "всплеска ошибки у стыка нет, склейка не создаёт заметного шва.")
        elif delta > 0:
            lines.append("Ошибка у границы выше. Формулировку в статье следует смягчить "
                         "до «контролируемый переходный слой» и привести этот профиль.")
        else:
            lines.append("Ошибка у границы НИЖЕ, чем в середине области. Скорее всего "
                         "дело не в стыке, а в рельефе: центр области горный.")

    (out_dir / "seam_profile.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ── данные для карты: срез через границу ──────────────────────────
    np.savez_compressed(
        out_dir / "seam_map_data.npz",
        lat=lat, lon=lon, is_regional=is_reg, dist_km=d,
        pred_last=pred[0, :, -1] * std, gt_last=gt[0, :, -1] * std,
        rmse_per_node=np.sqrt((err ** 2).mean(axis=(0, 2))),
        variable=a.var, n_samples=N,
    )
    print(f"\n→ {out_dir}/seam_profile.md")
    print(f"→ {out_dir}/seam_map_data.npz (для карты и профиля)")


if __name__ == "__main__":
    main()
