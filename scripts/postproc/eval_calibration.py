#!/usr/bin/env python3
"""Проверяет, соответствует ли заявленный разброс действительной ошибке.

Вероятностная настройка выдаёт не только поправку, но и σ — свою оценку того,
насколько она уверена. Точность от этого не страдает (2,289 против 2,297), то
есть оценка надёжности достаётся даром. Но даром она достаётся только если ей
можно верить: модель, систематически занижающая σ, будет уверенно ошибаться, а
завышающая — бесполезна, потому что «не знаю» она скажет всегда.

Что считается:

* СРЕДНЯЯ ОСТРОТА и СРЕДНЯЯ ОШИБКА. У правильно откалиброванной модели
  среднеквадратическая ошибка равна среднеквадратичному σ. Отношение больше
  единицы — самоуверенность, меньше — перестраховка.
* ПОКРЫТИЕ интервалов. В интервал ±1σ должно попадать 68,3 % случаев, в ±2σ —
  95,4 %. Отклонение показывает, где именно врёт оценка: в середине или в
  хвостах.
* НАДЁЖНОСТЬ ПО КОРЗИНАМ. Строки делятся по заявленному σ на десять корзин
  равного размера, и в каждой сравнивается заявленное с действительным. Средние
  величины могут сойтись случайно, при завышенной σ на лёгких случаях и
  заниженной на трудных; разбивка это ловит.
* НЕПРЕРЫВНАЯ ОЦЕНКА (CRPS) в сравнении с той же моделью, объявившей постоянную
  σ. Если постоянная не хуже — переменная σ бесполезна, сколь угодно
  откалиброванная.

    python3 scripts/postproc/eval_calibration.py --val-parquet T.parquet \\
        --ckpt experiments/.../best_model.pth --out-dir ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocessing.calibration import (coverage, crps_gaussian,
                                            reliability)
from src.postprocessing.neural.dataset import StationCorpusDataset
from src.postprocessing.neural.models import (StationLeadAwareResidualMLP,
                                              StationLeadBiasResidualMLP)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-parquet", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location=a.device, weights_only=False)
    cfg = ckpt["cfg"]
    if not cfg.get("probabilistic"):
        raise SystemExit(
            "модель точечная, разброса не выдаёт — калибровать нечего. "
            "Нужна настройка, обученная с --probabilistic.")

    ds = StationCorpusDataset(a.val_parquet, feature_cols=ckpt["feature_cols"],
                              scalers=ckpt["scalers"],
                              station_to_idx=ckpt["station_to_idx"],
                              auto_obs_features=False)
    cls = {"StationLeadAwareResidualMLP": StationLeadAwareResidualMLP,
           "StationLeadBiasResidualMLP": StationLeadBiasResidualMLP}[
        ckpt.get("model_class", "StationLeadBiasResidualMLP")]
    model = cls(feature_dim=len(ckpt["feature_cols"]),
                num_stations=len(ckpt["station_to_idx"]),
                station_emb_dim=cfg["station_emb_dim"], hidden=tuple(cfg["hidden"]),
                dropout=cfg["dropout"], probabilistic=True,
                film_hidden=cfg["film_hidden"]).to(a.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False)
    mu_all, sig_all = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["features"].to(a.device), b["station_idx"].to(a.device),
                        b["lead_norm"].to(a.device),
                        {k: b[f"gnn_{k}"].to(a.device) for k in ("t2m", "u10", "v10")})
            mu_all.append(out["t2m_mu"].cpu().numpy())
            sig_all.append(np.exp(out["t2m_log_sigma"].cpu().numpy()))
    mu = np.concatenate(mu_all)
    sigma = np.concatenate(sig_all)
    obs = ds.Y[:, 0]
    err = mu - obs

    rmse = float(np.sqrt((err ** 2).mean()))
    sharp = float(np.sqrt((sigma ** 2).mean()))
    cov1, cov2 = coverage(sigma, err, 1.0), coverage(sigma, err, 2.0)
    crps = float(crps_gaussian(mu, sigma, obs).mean())
    crps_const = float(crps_gaussian(mu, np.full_like(sigma, sharp), obs).mean())
    rel = reliability(sigma, err)

    print(f"[калибровка] строк {len(err):,}")
    print(f"  ошибка {rmse:.3f} °C, заявленный разброс {sharp:.3f} °C, "
          f"отношение {rmse / sharp:.3f}")
    print(f"    (больше единицы — самоуверенность, меньше — перестраховка)")
    print(f"  попадание в +-1 сигма: {cov1:.1f} % (должно быть 68,3)")
    print(f"  попадание в +-2 сигма: {cov2:.1f} % (должно быть 95,4)")
    print(f"  CRPS с переменной сигмой {crps:.4f}, с постоянной {crps_const:.4f} "
          f"({(crps_const - crps) / crps_const * 100:+.1f} % от переменной)")
    print("  надёжность по корзинам (заявлено -> действительно):")
    for r in rel:
        print(f"    n={r['n']:>7,}  сигма {r['sigma_mean']:6.3f} -> "
              f"ошибка {r['rmse']:6.3f}   {r['rmse'] / max(r['sigma_mean'], 1e-9):5.2f}x")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps({
        "n": len(err), "rmse": rmse, "sharpness": sharp, "ratio": rmse / sharp,
        "coverage_1sigma": cov1, "coverage_2sigma": cov2,
        "crps": crps, "crps_constant_sigma": crps_const,
        "reliability": rel}, ensure_ascii=False, indent=1))
    print(f"[калибровка] записано {out / 'calibration.json'}", flush=True)


if __name__ == "__main__":
    main()
