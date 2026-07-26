#!/usr/bin/env python3
"""Блочный бутстреп доверительных интервалов из per-sample метрик (*.npz).

Зачем: сроки начала прогноза идут с шагом 6 ч и сильно скоррелированы внутри
одного синоптического процесса. Обычный бутстреп по независимым срокам занижал бы
ширину интервала, поэтому ресэмплируются БЛОКИ подряд идущих сроков (по умолчанию
блок = 5 суток = 20 сроков).

Использование:
    # доверительный интервал для одного прогона
    python scripts/paper_bootstrap_ci.py docs/paper/runs/vm3_m1/m1_noda_ar28_samples.npz \
        --var t2m --scope region --horizons 4 8 16 20 28

    # интервал для РАЗНИЦЫ двух прогонов (парный, на общих сроках)
    python scripts/paper_bootstrap_ci.py A_samples.npz --vs B_samples.npz --var t2m --scope region
"""
import argparse
from pathlib import Path

import numpy as np


def load(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["variables"] = [str(v) for v in out["variables"]]
    return out


def rmse_phys(mse: np.ndarray, std: np.ndarray, ch: int) -> np.ndarray:
    """mse: (N, H, C) в стандартизованных единицах → RMSE канала ch в физических."""
    return np.sqrt(mse[:, :, ch]) * std[ch]


def block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Индексы одной бутстреп-реплики: склейка случайных блоков до длины n."""
    starts = rng.integers(0, max(n - block, 1), size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    return idx[:n]


def ci(values_per_sample: np.ndarray, block: int, reps: int, seed: int = 0):
    """values_per_sample: (N,) вклад каждого срока (квадраты ошибок). Возвращает (оценка, lo, hi)."""
    rng = np.random.default_rng(seed)
    n = values_per_sample.shape[0]
    point = np.sqrt(values_per_sample.mean())
    reps_vals = np.empty(reps)
    for r in range(reps):
        idx = block_indices(n, block, rng)
        reps_vals[r] = np.sqrt(values_per_sample[idx].mean())
    lo, hi = np.percentile(reps_vals, [2.5, 97.5])
    return point, lo, hi


def diff_ci(a: np.ndarray, b: np.ndarray, block: int, reps: int, seed: int = 0):
    """Парный интервал для разности RMSE(b) - RMSE(a) на общих сроках."""
    rng = np.random.default_rng(seed)
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    point = np.sqrt(b.mean()) - np.sqrt(a.mean())
    reps_vals = np.empty(reps)
    for r in range(reps):
        idx = block_indices(n, block, rng)
        reps_vals[r] = np.sqrt(b[idx].mean()) - np.sqrt(a[idx].mean())
    lo, hi = np.percentile(reps_vals, [2.5, 97.5])
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--vs", default=None, help="второй npz для парного сравнения")
    ap.add_argument("--var", default="t2m")
    ap.add_argument("--scope", default="region", choices=["region", "global"])
    ap.add_argument("--horizons", type=int, nargs="*", default=None,
                    help="номера шагов (1-based); по умолчанию все")
    ap.add_argument("--block", type=int, default=20, help="длина блока в сроках (20 = 5 суток)")
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()

    A = load(Path(a.npz))
    key = f"mse_pred_{a.scope}"
    if key not in A:
        raise SystemExit(f"в {a.npz} нет {key} (прогон без --region?)")
    ch = A["variables"].index(a.var)
    std = A["std"]
    H = A[key].shape[1]
    horizons = a.horizons or list(range(1, H + 1))

    B = load(Path(a.vs)) if a.vs else None
    unit = "°C" if a.var == "t2m" or a.var.startswith("t@") else ""

    print(f"# Бутстреп-ДИ (95%), блок {a.block} сроков ({a.block*6/24:.0f} сут), {a.reps} реплик")
    print(f"# файл: {Path(a.npz).name}" + (f"  против {Path(a.vs).name}" if B else ""))
    print(f"# переменная {a.var}, область {a.scope}, N={A[key].shape[0]} сроков\n")

    if B is None:
        print(f"| горизонт | RMSE, {unit or 'ед.'} | 95% ДИ |")
        print("|---|---:|---|")
        for h in horizons:
            v = A[key][:, h - 1, ch] * std[ch] ** 2   # (N,) квадраты ошибок в физ. ед.
            p, lo, hi = ci(v, a.block, a.reps)
            print(f"| +{h*6} ч | {p:.3f} | [{lo:.3f}, {hi:.3f}] |")
    else:
        keyB = f"mse_pred_{a.scope}"
        chB = B["variables"].index(a.var)
        print(f"| горизонт | A | B | разность B−A | 95% ДИ разности | значимо |")
        print("|---|---:|---:|---:|---|---|")
        for h in horizons:
            va = A[key][:, h - 1, ch] * std[ch] ** 2
            vb = B[keyB][:, h - 1, chB] * B["std"][chB] ** 2
            pa = np.sqrt(va.mean()); pb = np.sqrt(vb.mean())
            p, lo, hi = diff_ci(va, vb, a.block, a.reps)
            sig = "да" if (lo > 0 or hi < 0) else "нет"
            print(f"| +{h*6} ч | {pa:.3f} | {pb:.3f} | {p:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {sig} |")


if __name__ == "__main__":
    main()
