#!/usr/bin/env python3
"""Веса каналов в функции потерь по разбросу их шестичасовой невязки.

Зачем. Лосс — среднеквадратичная ошибка по нормированным каналам с равным весом.
Но нормировка делит на разброс САМОГО ПОЛЯ, а модель предсказывает ПРИРАЩЕНИЕ,
и разброс приращений у каналов отличается на порядки. В итоге доля канала в
целевой функции задана не нами, а этим отношением: осадки забирают около трети,
а приземная температура — те самые доли процента, ради которых пишется статья.

Скрипт считает по обучающей выборке σ_c — разброс невязки за 6 ч в нормированных
единицах — и выдаёт веса, выравнивающие вклад каналов:

    w_c = (σ_med / σ_c)^(2p),  p = 1 — полное выравнивание, 0.5 — половинное,

с ограничением сверху и снизу: у самых спокойных каналов (приземное давление)
невязка близка к шуму дискретизации, и полное выравнивание отдало бы им лосс.

Запуск:
    python scripts/channel_loss_weights.py --exp experiments/multires_krsk_33f \
        --data-dir /data/datasets/multires_krsk_33f --out weights.json
"""
import argparse, json, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataloader_chunked import TimeseriesChunkDataset

NAMES = ['t2m','10u','10v','msl','tp','sp','tcwv','z_surf','lsm',
         't@850','u@850','v@850','z@850','q@850','t@500','u@500','v@500','z@500','q@500',
         'z@250','t@250','u@250','v@250','q@250',
         'z@1000','t@1000','u@1000','v@1000','q@1000',
         'sin_hour','cos_hour','sin_doy','cos_doy']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, help='каталог опыта с config.json')
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--samples', type=int, default=400, help='сколько сроков усреднить')
    ap.add_argument('--power', type=float, default=1.0, help='1 — полное выравнивание')
    ap.add_argument('--cap', type=float, default=25.0, help='предел веса сверху и снизу')
    ap.add_argument('--boost', type=float, default=1.0,
                    help='множитель к весам публикуемых каналов поверх выравнивания')
    ap.add_argument('--boost-channels', default='t2m,10u,10v,msl,z@500',
                    help='каким каналам множитель (имена через запятую)')
    ap.add_argument('--out', default=None)
    # Те же σ годятся и на шум в авторегрессии: шум должен быть долей того,
    # что канал реально меняет за 6 ч, а не одинаковым для всех.
    ap.add_argument('--out-noise', default=None, help='куда записать сигмы шума по каналам')
    ap.add_argument('--noise-k', type=float, default=0.3,
                    help='доля от σ невязки, которая идёт в шум (по умолчанию 0.3)')
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.exp, 'config.json')))
    n_feat = cfg['data']['num_features_used']
    no_loss = set(cfg.get('static_channels', []) or []) | set(cfg.get('forcing_channels', []) or [])

    ds = TimeseriesChunkDataset(a.data_dir, obs_window=2, pred_steps=1,
                                split='train', n_features=n_feat)
    print(f"обучающая выборка: {len(ds)} сроков, каналов {n_feat}, "
          f"вне лосса {sorted(no_loss)}")

    # Невязка за 6 ч в нормированных единицах: последний входной кадр минус цель.
    # Копим сумму квадратов, а не массив — узлов 133 тысячи на срок.
    idx = np.linspace(0, len(ds) - 1, min(a.samples, len(ds))).astype(int)
    ssq = np.zeros(n_feat, dtype=np.float64)
    cnt = 0
    for k, i in enumerate(idx):
        X, Y = ds[int(i)]
        X = X.numpy().reshape(-1, 2, n_feat)      # (N, obs, feat)
        Y = Y.numpy().reshape(-1, 1, n_feat)      # (N, pred, feat)
        d = Y[:, 0, :] - X[:, -1, :]              # (N, feat)
        ssq += np.sum(d.astype(np.float64) ** 2, axis=0)
        cnt += d.shape[0]
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(idx)} сроков")
    sigma = np.sqrt(ssq / cnt)

    dyn = [c for c in range(n_feat) if c not in no_loss]
    # Медиана как опорная точка: веса выходят около единицы, и общий масштаб
    # лосса не уезжает — темп обучения можно не трогать.
    sig_med = float(np.median(sigma[dyn]))
    # Множитель применяется ДО ограничения: иначе канал, упёршийся в потолок
    # на выравнивании, множитель просто проигнорировал бы.
    boost_idx = set()
    if a.boost != 1.0:
        for nm in a.boost_channels.split(','):
            nm = nm.strip()
            if nm in NAMES and NAMES.index(nm) in dyn:
                boost_idx.add(NAMES.index(nm))
            elif nm:
                print(f"[!] канал '{nm}' не найден или вне лосса — пропускаю")
    w = {}
    for c in dyn:
        if sigma[c] <= 0:
            continue
        wc = (sig_med / sigma[c]) ** (2 * a.power)
        if c in boost_idx:
            wc *= a.boost
        w[str(c)] = float(np.clip(wc, 1.0 / a.cap, a.cap))
    if boost_idx:
        print(f"множитель {a.boost:g} применён к каналам: "
              + ", ".join(NAMES[c] for c in sorted(boost_idx)))

    share_old = {c: sigma[c] ** 2 for c in dyn}
    share_new = {c: w[str(c)] * sigma[c] ** 2 for c in dyn}
    tot_o, tot_n = sum(share_old.values()), sum(share_new.values())

    print(f"\nσ медианный {sig_med:.5f}, степень {a.power}, предел {a.cap}\n")
    print(f"{'канал':>9} {'σ невязки':>10} {'доля было':>10} {'вес':>8} {'доля стало':>11}")
    for c in sorted(dyn, key=lambda c: -share_old[c]):
        nm = NAMES[c] if c < len(NAMES) else f'ch{c}'
        print(f"{nm:>9} {sigma[c]:>10.5f} {share_old[c]/tot_o*100:>9.2f}% "
              f"{w[str(c)]:>8.3f} {share_new[c]/tot_n*100:>10.2f}%")

    pub = ['t2m', '10u', '10v', 'msl', 'z@500']
    pi = [NAMES.index(p) for p in pub if NAMES.index(p) in dyn]
    print(f"\nпять каналов из таблиц статьи: было "
          f"{sum(share_old[c] for c in pi)/tot_o*100:.1f} %, стало "
          f"{sum(share_new[c] for c in pi)/tot_n*100:.1f} %")
    print(f"самый тяжёлый канал: было {max(share_old.values())/tot_o*100:.1f} %, "
          f"стало {max(share_new.values())/tot_n*100:.1f} %")

    if a.out:
        json.dump(w, open(a.out, 'w'), indent=2)
        print(f"\nвеса записаны в {a.out}")

    if a.out_noise:
        ns = {str(c): float(a.noise_k * sigma[c]) for c in dyn if sigma[c] > 0}
        json.dump(ns, open(a.out_noise, 'w'), indent=2)
        vals = sorted(ns.values())
        print(f"\nсигмы шума (k={a.noise_k}) записаны в {a.out_noise}")
        print(f"  диапазон {vals[0]:.5f} … {vals[-1]:.5f}, медиана {vals[len(vals)//2]:.5f}")
        print(f"  для сравнения, единая сигма 0,05 из прогона 22.08 была бы для "
              f"самого спокойного канала в {0.05/vals[0]:.0f} раз больше нужной")


if __name__ == '__main__':
    main()
