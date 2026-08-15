#!/usr/bin/env python3
"""Климатологический эталон для статьи.

Что это. Прогноз «как обычно бывает в это время года»: среднее многолетнее для
данного узла, дня года и часа. Начальное состояние не используется вовсе.

Зачем. Инерционный эталон на дальних сроках насыщается и перестаёт быть строгим
(замер 15.08.2026: приземная температура, инерция 2,43 °C на сутках и всего
4,96 °C на двух неделях). Климатология, наоборот, от горизонта не зависит
совсем — её ошибка одинакова на +6 ч и на +336 ч. Поэтому она задаёт предел
полезности прогноза: горизонт, на котором модель с ней сравнивается, и есть
граница, за которой смотреть в справочник не хуже.

Как считается. Климатология приближается гармониками по дню года (3 гармоники)
и по часу суток (2 гармоники) — итого 11 коэффициентов на узел и канал. Это
устойчивее поячейкового среднего: при 10 годах на каждый срок приходится всего
десяток значений, и «средние» получаются шумными.

Обучается СТРОГО на обучающей части выборки, тестовые сроки в подгонку не
попадают — иначе эталон подглядывает в ответ.

Сравнение ведётся на тех же сроках, что и модель: индексы берутся из
`--samples` (файл, сохранённый predict.py через --save-sample-metrics), поэтому
числа сопоставимы построчно.

Запуск:
  python scripts/paper_climatology.py \
      --data-dir /data/datasets/multires_krsk_33f \
      --samples /workdir/paper_results/m33_last_roi_samples.npz \
      --out /workdir/paper_results/clim_krsk.npz
"""
import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np

N_DOY_HARM = 3
# Суточный ход: косинус и синус первой гармоники плюс ТОЛЬКО косинус второй.
# Данные шестичасовые, часы бывают лишь 0, 6, 12, 18 — а sin(2*2pi*h/24) на них
# тождественно ноль, то есть столбец матрицы плана вырожден. По четырём срокам в
# сутках восстановимы ровно четыре параметра: константа, cos1, sin1, cos2.
N_COEF = 1 + 2 * N_DOY_HARM + 3  # 10


def design_row(dt: datetime) -> np.ndarray:
    """Строка матрицы плана для момента времени."""
    doy = dt.timetuple().tm_yday
    hour = dt.hour + dt.minute / 60.0
    row = [1.0]
    for k in range(1, N_DOY_HARM + 1):
        a = 2 * np.pi * k * doy / 365.25
        row += [np.cos(a), np.sin(a)]
    a1 = 2 * np.pi * hour / 24.0
    row += [np.cos(a1), np.sin(a1), np.cos(2 * a1)]
    return np.asarray(row, dtype=np.float64)


def open_dataset(data_dir: str):
    """Открывает memmap датасета, возвращает (массив, info, mean, std)."""
    info = json.load(open(os.path.join(data_dir, "dataset_info.json")))
    sc = np.load(os.path.join(data_dir, "scalers.npz"))
    mean = sc["mean"].astype(np.float32)
    std = sc["std"].astype(np.float32)

    n_base = info.get("n_feat_base", info["n_feat"])
    if info.get("flat", False):
        shape = (info["n_time"], info["n_nodes"], n_base)
    else:
        shape = (info["n_time"], info["n_lon"], info["n_lat"], n_base)
    base = np.memmap(os.path.join(data_dir, "data.npy"), dtype=np.float16,
                     mode="r", shape=shape)

    extra = None
    if info.get("extra_file"):
        n_extra = info["n_feat_extra"]
        eshape = shape[:-1] + (n_extra,)
        extra = np.memmap(os.path.join(data_dir, info["extra_file"]),
                          dtype=np.float16, mode="r", shape=eshape)
    return base, extra, info, mean, std


def frame(base, extra, i: int, n_feat: int) -> np.ndarray:
    """Один срок, нормированный, форма (узлы, каналы)."""
    a = base[i]
    if extra is not None:
        a = np.concatenate([np.asarray(a), np.asarray(extra[i])], axis=-1)
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 3:  # (lon, lat, C) -> lat-major, как в даталоадере
        a = a.transpose(1, 0, 2)
    return a.reshape(-1, a.shape[-1])[:, :n_feat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--samples", required=True,
                    help="npz от predict.py --save-sample-metrics: берём t_offset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs-window", type=int, default=2)
    ap.add_argument("--fit-day-stride", type=int, default=5,
                    help="каждые N суток обучающей части идут в подгонку ЦЕЛИКОМ, все "
                         "четыре срока. Равномерное прореживание по срокам не годится: "
                         "шаг в 42 ч не разрешает суточный ход, часы сцепляются с днями "
                         "и матрица плана вырождается (проверено 15.08.2026)")
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--region", nargs=4, type=float, default=None,
                    metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    args = ap.parse_args()

    base, extra, info, mean, std = open_dataset(args.data_dir)
    n_feat = info["n_feat"]
    T = info["n_time"]
    t0 = datetime.fromisoformat(info["time_start"])
    print(f"[clim] датасет {T} сроков от {t0.date()}, каналов {n_feat}")

    z = np.load(args.samples)
    t_off = z["t_offset"].astype(int)
    ar_steps = int(z["ar_steps"])
    print(f"[clim] сроков для оценки {len(t_off)}, горизонтов {ar_steps}")

    # --- узлы области ---
    reg_idx = None
    if args.region is not None:
        co = np.load(os.path.join(args.data_dir, "coords.npz"))
        lat, lon = co["latitude"], co["longitude"]
        if lat.ndim == 1 and len(lat) != frame(base, extra, 0, n_feat).shape[0]:
            # регулярная сетка: собираем пары lat-major
            lo, la = np.meshgrid(lon, lat)
            lat, lon = la.ravel(), lo.ravel()
        m = ((lat >= args.region[0]) & (lat <= args.region[1]) &
             (lon >= args.region[2]) & (lon <= args.region[3]))
        reg_idx = np.where(m)[0]
        print(f"[clim] узлов в области: {len(reg_idx)}")

    # --- граница обучающей части ---
    # Подгонка ТОЛЬКО по обучающим срокам: тестовые в неё попадать не должны.
    n_samples_total = T - (args.obs_window + ar_steps) + 1
    split_idx = int(n_samples_total * (1 - args.test_fraction))
    t_train_end = split_idx  # срок, начиная с которого идут проверочная и тестовая
    print(f"[clim] подгонка по срокам [0, {t_train_end}) — это до "
          f"{(t0 + timedelta(hours=6 * t_train_end)).date()}")

    # --- накопление нормальных уравнений ---
    # Каждые N суток берём целиком: суточный ход должен быть разрешён.
    days = np.arange(0, t_train_end // 4, args.fit_day_stride)
    fit_idx = (days[:, None] * 4 + np.arange(4)[None, :]).ravel()
    fit_idx = fit_idx[fit_idx < t_train_end]
    N_nodes = frame(base, extra, 0, n_feat).shape[0]
    XtX = np.zeros((N_COEF, N_COEF), dtype=np.float64)
    XtY = np.zeros((N_COEF, N_nodes, n_feat), dtype=np.float32)
    for n, i in enumerate(fit_idx):
        x = design_row(t0 + timedelta(hours=6 * int(i)))
        y = (frame(base, extra, int(i), n_feat) - mean) / std
        XtX += np.outer(x, x)
        XtY += x.astype(np.float32)[:, None, None] * y[None]
        if (n + 1) % 200 == 0:
            print(f"[clim] подгонка {n + 1}/{len(fit_idx)}")
    coef = np.linalg.solve(XtX, XtY.reshape(N_COEF, -1).astype(np.float64))
    coef = coef.reshape(N_COEF, N_nodes, n_feat).astype(np.float32)
    print(f"[clim] коэффициенты найдены по {len(fit_idx)} срокам")

    # --- оценка на тех же сроках, что и модель ---
    mse_g = np.zeros((len(t_off), ar_steps, n_feat), dtype=np.float64)
    mse_r = np.zeros((len(t_off), ar_steps, n_feat), dtype=np.float64) if reg_idx is not None else None
    for s, ts in enumerate(t_off):
        for h in range(ar_steps):
            # окно даталоадера: obs-кадры [ts .. ts+obs-1], цели дальше подряд
            fi = int(ts) + args.obs_window + h
            if fi >= T:
                continue
            x = design_row(t0 + timedelta(hours=6 * fi))
            pred = np.tensordot(x.astype(np.float32), coef, axes=(0, 0))
            truth = (frame(base, extra, fi, n_feat) - mean) / std
            d2 = (pred - truth) ** 2
            mse_g[s, h] = d2.mean(axis=0)
            if reg_idx is not None:
                mse_r[s, h] = d2[reg_idx].mean(axis=0)
        if (s + 1) % 200 == 0:
            print(f"[clim] оценка {s + 1}/{len(t_off)}")

    out = {"mse_clim_global": mse_g, "t_offset": t_off,
           "ar_steps": ar_steps, "std": std}
    if mse_r is not None:
        out["mse_clim_region"] = mse_r
    np.savez_compressed(args.out, **out)
    print(f"[clim] сохранено → {args.out}")

    # --- сводка ---
    names = json.load(open(os.path.join(args.data_dir, "variables.json")))
    show = [n for n in ("t2m", "msl", "10u", "10v", "t@850") if n in names]
    tag, mm = ("область", mse_r) if mse_r is not None else ("вся сетка", mse_g)
    print(f"\nКлиматологический эталон, {tag}, физические единицы:")
    print(f"{'канал':>8} " + " ".join(f"{'+' + str(6 * (h + 1)) + 'ч':>9}" for h in range(ar_steps)))
    for nm in show:
        c = names.index(nm)
        row = [np.sqrt(mm[:, h, c].mean()) * std[c] for h in range(ar_steps)]
        print(f"{nm:>8} " + " ".join(f"{v:9.2f}" for v in row))

    if "mse_pred_region" in z or "mse_pred_global" in z:
        key = "mse_pred_region" if (mse_r is not None and "mse_pred_region" in z) else "mse_pred_global"
        mp = z[key]
        print(f"\nУспешность модели относительно климатологии (%), {tag}:")
        print(f"{'канал':>8} " + " ".join(f"{'+' + str(6 * (h + 1)) + 'ч':>9}" for h in range(ar_steps)))
        for nm in show:
            c = names.index(nm)
            row = []
            for h in range(ar_steps):
                rm = np.sqrt(mp[:, h, c].mean())
                rc = np.sqrt(mm[:, h, c].mean())
                row.append(100 * (1 - rm / rc) if rc > 0 else float("nan"))
            print(f"{nm:>8} " + " ".join(f"{v:9.1f}" for v in row))
        print("\nПоложительное значение — модель лучше климатологии.")
        print("Отрицательное — на этом горизонте прогноз уже бесполезен.")


if __name__ == "__main__":
    main()
