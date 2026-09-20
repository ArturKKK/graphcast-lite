#!/usr/bin/env bash
# Целы ли сохранённые посрочные метрики и есть ли в них слагаемые ACC.
#
# Нужно после сбоя или двойного запуска: два процесса, пишущие один npz,
# оставляют обрезанный файл, а np.load падает только при чтении — то есть
# через час счёта, а не сразу.
#
# Запуск:  bash scripts/_vm_check_npz.sh
set -uo pipefail
OUT=${OUT:-/workdir/paper_results}
python3 - "$OUT" <<'PY'
import sys, pathlib
import numpy as np
out = pathlib.Path(sys.argv[1])
files = sorted(out.glob("w_*_samples.npz"))
if not files:
    print("файлов w_*_samples.npz нет"); raise SystemExit(0)
NEED = ("acc_num_region", "acc_ff_region", "acc_aa_region")
bad = []
for f in files:
    try:
        d = np.load(f, allow_pickle=True)
        keys = set(d.files)
        n = d["mse_pred_region"].shape[0] if "mse_pred_region" in keys else 0
        miss = [k for k in NEED if k not in keys]
        used = int((d["t_offset"] >= 0).sum()) if "t_offset" in keys else -1
        status = "ок"
        if miss:
            status = f"НЕТ слагаемых ACC: {miss}"; bad.append(f)
        elif used < n:
            status = f"заполнено {used} из {n} сроков — прогон не доехал"; bad.append(f)
        print(f"  {f.name:32s} сроков {n:5d}, заполнено {used:5d}  {status}")
    except Exception as e:
        print(f"  {f.name:32s} ЧИТАЕТСЯ С ОШИБКОЙ: {type(e).__name__}: {e}")
        bad.append(f)
print()
if bad:
    print("НЕГОДНЫЕ — удалить и пересчитать:")
    for f in bad:
        print(f"  rm {f}")
else:
    print("все файлы целые, слагаемые ACC на месте")
PY
