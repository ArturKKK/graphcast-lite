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


# Каналы, не входящие в агрегат: статические поля и временной форсинг. Список
# повторяет то, что делает predict.py (no_loss_ch), иначе агрегат отсюда не
# совпал бы с публикуемым.
STATIC_FORCING = {"z_surf", "lsm", "sin_hour", "cos_hour", "sin_doy", "cos_doy"}


def dynamic_channels(variables) -> list:
    return [i for i, v in enumerate(variables) if v not in STATIC_FORCING]


def agg_terms(d: dict, scope: str, horizons: list) -> tuple:
    """Посрочные вклады в агрегатную ошибку прогноза и эталона.

    Агрегат считается в НОРМИРОВАННЫХ единицах по всем динамическим каналам и
    горизонтам сразу — так же, как StreamingMetrics в predict.py. Возвращаем два
    массива (N,): средний квадрат ошибки по каналам и горизонтам для каждого
    срока, отдельно для прогноза и для инерционного эталона.
    """
    ch = dynamic_channels(d["variables"])
    hs = [h - 1 for h in horizons]
    pred = d[f"mse_pred_{scope}"][:, hs][:, :, ch].mean(axis=(1, 2))
    base = d[f"mse_base_{scope}"][:, hs][:, :, ch].mean(axis=(1, 2))
    return pred, base


def skill(pred: np.ndarray, base: np.ndarray) -> float:
    """Агрегатная успешность в процентах по формуле (3)."""
    return (1.0 - np.sqrt(pred.mean()) / np.sqrt(base.mean())) * 100.0


def skill_ci(pred, base, block, reps, seed=0):
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    point = skill(pred, base)
    vals = np.empty(reps)
    for r in range(reps):
        idx = block_indices(n, block, rng)
        vals[r] = skill(pred[idx], base[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def skill_diff_ci(pa, ba, pb, bb, block, reps, seed=0):
    """Парный интервал для разности успешностей S(B) - S(A) на общих сроках."""
    rng = np.random.default_rng(seed)
    n = min(pa.shape[0], pb.shape[0])
    pa, ba, pb, bb = pa[:n], ba[:n], pb[:n], bb[:n]
    point = skill(pb, bb) - skill(pa, ba)
    vals = np.empty(reps)
    for r in range(reps):
        idx = block_indices(n, block, rng)
        vals[r] = skill(pb[idx], bb[idx]) - skill(pa[idx], ba[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def acc_terms(d: dict, scope: str, horizons: list, ch: int) -> tuple:
    """Посрочные слагаемые ACC одного канала: (числитель, ff, aa).

    ACC — отношение сумм, а не среднее корреляций, поэтому реплика бутстрепа
    обязана пересобрать суммы по своим срокам и поделить уже их.
    """
    need = [f"acc_{k}_{scope}" for k in ("num", "ff", "aa")]
    missing = [k for k in need if k not in d]
    if missing:
        raise SystemExit(
            f"в прогоне нет {missing}: он считался без --climatology, "
            f"и ACC из него не восстановить")
    hs = [h - 1 for h in horizons]
    return tuple(d[k][:, hs, ch].sum(axis=1) for k in need)


def acc(num, ff, aa) -> float:
    den = np.sqrt(ff.sum() * aa.sum())
    return float(num.sum() / den) if den > 0 else float("nan")


def acc_ci(num, ff, aa, block, reps, seed=0):
    rng = np.random.default_rng(seed)
    n = num.shape[0]
    point = acc(num, ff, aa)
    vals = np.empty(reps)
    for r in range(reps):
        idx = block_indices(n, block, rng)
        vals[r] = acc(num[idx], ff[idx], aa[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--vs", default=None, help="второй npz для парного сравнения")
    ap.add_argument("--var", default="t2m",
                    help="имя канала либо 'aggregate' — агрегатная успешность")
    ap.add_argument("--scope", default="region", choices=["region", "global"])
    ap.add_argument("--horizons", type=int, nargs="*", default=None,
                    help="номера шагов (1-based); по умолчанию все")
    ap.add_argument("--acc", action="store_true",
                    help="считать ACC против климатологии вместо RMSE "
                         "(прогон должен быть сделан с --climatology)")
    ap.add_argument("--block", type=int, default=20, help="длина блока в сроках (20 = 5 суток)")
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()

    A = load(Path(a.npz))
    key = f"mse_pred_{a.scope}"
    if key not in A:
        raise SystemExit(f"в {a.npz} нет {key} (прогон без --region?)")

    if a.var == "aggregate":
        H_all = A[key].shape[1]
        hs = a.horizons or list(range(1, H_all + 1))
        pa, ba = agg_terms(A, a.scope, hs)
        print(f"# Бутстреп-ДИ (95%), блок {a.block} сроков "
              f"({a.block*6/24:.0f} сут), {a.reps} реплик")
        print(f"# файл: {Path(a.npz).name}"
              + (f"  против {Path(a.vs).name}" if a.vs else ""))
        print(f"# агрегатная успешность, область {a.scope}, "
              f"каналов {len(dynamic_channels(A['variables']))}, "
              f"горизонтов {len(hs)}, N={len(pa)} сроков\n")
        if a.vs is None:
            p, lo, hi = skill_ci(pa, ba, a.block, a.reps)
            print("| величина | оценка | 95% ДИ |")
            print("|---|---:|---|")
            print(f"| S, % | {p:.2f} | [{lo:.2f}, {hi:.2f}] |")
        else:
            B = load(Path(a.vs))
            pb, bb = agg_terms(B, a.scope, hs)
            sa, sb = skill(pa, ba), skill(pb, bb)
            p, lo, hi = skill_diff_ci(pa, ba, pb, bb, a.block, a.reps)
            sign = "ДА" if (lo > 0) == (hi > 0) else "нет"
            print("| A, % | B, % | разность B−A, п.п. | 95% ДИ | значимо |")
            print("|---:|---:|---:|---|---|")
            print(f"| {sa:.2f} | {sb:.2f} | {p:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {sign} |")
        return

    ch = A["variables"].index(a.var)
    std = A["std"]

    if a.acc:
        H_all = A[key].shape[1]
        hs = a.horizons or list(range(1, H_all + 1))
        print(f"# Бутстреп-ДИ (95%), блок {a.block} сроков, {a.reps} реплик")
        print(f"# файл: {Path(a.npz).name}, канал {a.var}, область {a.scope}\n")
        print("| горизонт | ACC | 95% ДИ |")
        print("|---|---:|---|")
        for h in hs:
            n_, f_, aa_ = acc_terms(A, a.scope, [h], ch)
            pnt, lo, hi = acc_ci(n_, f_, aa_, a.block, a.reps)
            print(f"| +{h*6} ч | {pnt:.4f} | [{lo:.4f}, {hi:.4f}] |")
        n_, f_, aa_ = acc_terms(A, a.scope, hs, ch)
        pnt, lo, hi = acc_ci(n_, f_, aa_, a.block, a.reps)
        print(f"| все | {pnt:.4f} | [{lo:.4f}, {hi:.4f}] |")
        return
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
