#!/usr/bin/env python3
"""Проверка, что собранный датасет действительно дособран.

Зачем. Файл создаётся memmap'ом сразу нужного размера и заполнен нулями, так
что и размер файла, и его наличие ничего не доказывают: прерванная загрузка
оставляет файл правильного размера с нулями в хвосте. 23.08.2026 на этом чуть
не упаковали и не удалили недокачанную порцию.

Что проверяется:
  1. builder объявил завершение: есть dataset_info.json, нет progress.json
     (скрипт сборки удаляет progress.json последним действием);
  2. размер data.npy совпадает с размерностями из dataset_info.json;
  3. ни один срок не состоит целиком из нулей — именно так выглядит
     недописанный хвост (запись идёт блоками по времени, все каналы сразу);
  4. в scalers.npz нет NaN и нулевых стандартных отклонений.

Запуск:
    python3 scripts/verify_dataset.py <каталог> [ещё каталоги...]
Код возврата 0 — всё хорошо, 1 — есть замечания.
"""
import json, sys, pathlib
import numpy as np

DT = {"float16": np.float16, "float32": np.float32}


def check(d: pathlib.Path) -> list:
    bad = []
    info_p, prog_p = d / "dataset_info.json", d / "progress.json"

    if not info_p.exists():
        return [f"нет dataset_info.json — сборка не доходила до конца"]
    if prog_p.exists():
        try:
            pr = json.loads(prog_p.read_text())
            bad.append(f"остался progress.json (докачано до срока "
                       f"{pr.get('last_completed_timestep')}) — сборка НЕ завершена")
        except Exception:
            bad.append("остался progress.json — сборка НЕ завершена")

    info = json.loads(info_p.read_text())
    n_t, n_lon, n_lat, n_f = (int(info[k]) for k in ("n_time", "n_lon", "n_lat", "n_feat"))
    dt = DT.get(info.get("dtype", "float16"), np.float16)
    data_p = d / info.get("file", "data.npy")
    if not data_p.exists():
        return bad + [f"нет {data_p.name}"]

    want = n_t * n_lon * n_lat * n_f * np.dtype(dt).itemsize
    got = data_p.stat().st_size
    if got != want:
        bad.append(f"размер {got} байт, ожидался {want} "
                   f"({n_t}×{n_lon}×{n_lat}×{n_f}×{np.dtype(dt).itemsize})")
        return bad

    fp = np.memmap(data_p, dtype=dt, mode="r", shape=(n_t, n_lon, n_lat, n_f))
    # Хвост важнее всего: если прервали, нули будут именно там. Плюс равномерная
    # выборка по всему файлу — на случай дыры в середине.
    probe = sorted(set(
        list(range(max(0, n_t - 8), n_t)) +
        [int(x) for x in np.linspace(0, n_t - 1, 40)]
    ))
    zero_t = [t for t in probe if not fp[t].any()]
    if zero_t:
        bad.append(f"нулевых сроков среди проверенных {len(zero_t)} из {len(probe)}: "
                   f"{zero_t[:6]}{'…' if len(zero_t) > 6 else ''} — данные не дописаны")
    else:
        # Отдельно каналы: канал, который нигде не записан, тоже дыра.
        sl = fp[probe[::4]]
        dead = [i for i in range(n_f) if not np.asarray(sl[..., i]).any()]
        if dead:
            names = info.get("variables", [])
            bad.append("каналы всюду нулевые: " +
                       ", ".join(names[i] if i < len(names) else str(i) for i in dead))

    sc_p = d / "scalers.npz"
    if not sc_p.exists():
        bad.append("нет scalers.npz — статистика не посчитана")
    else:
        sc = np.load(sc_p)
        if "std" in sc:
            std = np.asarray(sc["std"], dtype=np.float64)
            if not np.isfinite(std).all():
                bad.append("в scalers.npz есть NaN или бесконечность")
            n_zero = int((std == 0).sum())
            if n_zero:
                bad.append(f"нулевой разброс у {n_zero} каналов "
                           f"(нормировка на них даст деление на ноль)")
    return bad


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    rc = 0
    for arg in sys.argv[1:]:
        d = pathlib.Path(arg)
        print(f"\n=== {d} ===")
        if not d.is_dir():
            print("  каталога нет"); rc = 1; continue
        try:
            bad = check(d)
        except Exception as e:
            print(f"  проверка сорвалась: {e}"); rc = 1; continue
        if bad:
            rc = 1
            for b in bad:
                print(f"  ✗ {b}")
        else:
            info = json.loads((d / "dataset_info.json").read_text())
            print(f"  ✓ дособран: {info['n_time']} сроков × {info['n_feat']} каналов, "
                  f"{info.get('time_start')} — {info.get('time_end')}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
