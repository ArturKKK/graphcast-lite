#!/usr/bin/env python3
"""Метаданные 33-канального multires-датасета без самого датасета.

Сборщику корпуса из всего multires-каталога нужны только три файла: coords.npz,
scalers.npz и variables.json. Сам массив data.npy он не читает — кадры собираются
на лету из исходных датасетов. Поэтому строить 33-канальный датасет целиком
(около 56 ГБ и полчаса) ради трёх файлов незачем, тем более что вместе со слитым
источником на 81 ГБ это упирается в лимит диска платформы.

Состав повторяет build_multires_russia_33f.py дословно:
  coords.npz    — копия из слитого 19-канального источника (порядок узлов тот же);
  scalers.npz   — склейка (19,) из слитого + (10,) из ГЛОБАЛЬНОГО extra +
                  (4,) для временного форсинга: sin/cos равномерны, среднее 0,
                  стандартное отклонение 1/sqrt(2);
  variables.json — 33 имени в порядке слоёв.

Запуск:
    python3 scripts/postproc/make_multires33f_meta.py \
        --merged  /data/datasets/multires_krsk_19f_merge \
        --extra   /data/datasets/global_512x256_extra_2010-2021_07deg \
        --out     /data/datasets/multires_krsk_33f_meta
"""
import argparse, json, shutil
from pathlib import Path

import numpy as np

BASE_VARS = ["t2m", "10u", "10v", "msl", "tp", "sp", "tcwv", "z_surf", "lsm",
             "t@850", "u@850", "v@850", "z@850", "q@850",
             "t@500", "u@500", "v@500", "z@500", "q@500"]
EXTRA_PLEV = ["z@250", "t@250", "u@250", "v@250", "q@250",
              "z@1000", "t@1000", "u@1000", "v@1000", "q@1000"]
TIME_VARS = ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]
ALL_VARS = BASE_VARS + EXTRA_PLEV + TIME_VARS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="слитый multires 19f")
    ap.add_argument("--extra", required=True, help="ГЛОБАЛЬНЫЙ extra (источник нормировок plev)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    merged, extra, out = Path(a.merged), Path(a.extra), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    assert len(ALL_VARS) == 33, len(ALL_VARS)

    # 1. координаты
    src_coords = merged / "coords.npz"
    if not src_coords.exists():
        raise SystemExit(f"нет {src_coords}")
    shutil.copy(src_coords, out / "coords.npz")
    c = np.load(out / "coords.npz")
    n_nodes = len(c["latitude"])
    has_reg = "is_regional" in c.files
    n_reg = int(c["is_regional"].sum()) if has_reg else -1
    print(f"[coords] {n_nodes} узлов, is_regional {'есть' if has_reg else 'НЕТ'}"
          + (f", из них региональных {n_reg}" if has_reg else ""))
    if not has_reg:
        raise SystemExit("в coords.npz нет is_regional — сборщик не сможет "
                         "развести узлы вставки и глобальной части")

    # 2. нормировки
    m = np.load(merged / "scalers.npz")
    m_mean, m_std = m["mean"].astype(np.float32), m["std"].astype(np.float32)
    if m_mean.shape != (19,):
        raise SystemExit(f"в {merged}/scalers.npz {m_mean.shape}, ожидалось (19,)")
    ext_path = extra / "scalers_extra.npz"
    if not ext_path.exists():
        ext_path = extra / "scalers.npz"
    e = np.load(ext_path)
    e_mean, e_std = e["mean"].astype(np.float32), e["std"].astype(np.float32)
    if e_mean.shape != (10,):
        raise SystemExit(f"в {ext_path} {e_mean.shape}, ожидалось (10,)")
    # sin/cos равномерны на окружности: среднее 0, дисперсия 1/2
    t_mean = np.zeros(4, dtype=np.float32)
    t_std = np.full(4, 1 / np.sqrt(2), dtype=np.float32)

    all_mean = np.concatenate([m_mean, e_mean, t_mean])
    all_std = np.concatenate([m_std, e_std, t_std])
    assert all_mean.shape == (33,) and all_std.shape == (33,)
    if not np.all(np.isfinite(all_mean)) or not np.all(np.isfinite(all_std)):
        raise SystemExit("в нормировках есть NaN или бесконечность")
    if np.any(all_std <= 0):
        bad = [ALL_VARS[i] for i in np.where(all_std <= 0)[0]]
        raise SystemExit(f"нулевое стандартное отклонение у каналов: {bad}")
    np.savez(out / "scalers.npz", mean=all_mean, std=all_std)
    print(f"[scalers] (33,) собраны: 19 из слитого + 10 из {ext_path.name} + 4 временных")

    # 3. имена каналов
    (out / "variables.json").write_text(json.dumps(ALL_VARS, indent=2))
    print(f"[variables] 33 имени записаны")

    # 4. сводка для глаза — числа должны быть физически осмысленными
    print(f"\n{'канал':>10} {'среднее':>12} {'ст.откл.':>10}")
    for i in (0, 3, 9, 17, 19, 24, 29):
        print(f"{ALL_VARS[i]:>10} {all_mean[i]:12.3f} {all_std[i]:10.3f}")
    print(f"\nготово: {out}")


if __name__ == "__main__":
    main()
