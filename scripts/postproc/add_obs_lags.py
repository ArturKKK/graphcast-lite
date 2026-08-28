#!/usr/bin/env python3
"""Добавляет к корпусу наблюдения станции, известные к моменту выпуска прогноза.

Зачем. Все 26 признаков корпуса описывают атмосферу ПО МНЕНИЮ МОДЕЛИ. Ни один не
говорит, что станция реально намерила перед выпуском прогноза. А именно это
различает режимы: в антициклон зимой инверсия даёт +10 °C расхождения, в адвекцию
+1 °C, и оба случая — «январь, 03 UTC», то есть таблица их не различает никак.

Главное про утечку. Признаки строятся ТОЛЬКО из наблюдений на момент выпуска
прогноза и раньше: obs(t0), obs(t0-6ч), obs(t0-12ч), obs(t0-24ч). Брать
наблюдение на срок действия прогноза нельзя — при сроке +24 ч это была бы
подсказка из будущего, и модель бы «научилась» тому, чего в работе не будет.
Скрипт это проверяет явно: все использованные метки времени обязаны быть не
позже времени выпуска.

Что добавляется:
    obs_t2m_lag0/6/12/24   — наблюдения к моменту выпуска, °C
    obs_t2m_tend24         — изменение за сутки перед выпуском
    obs_t2m_anom           — отклонение от климатической нормы станции для
                             этого месяца и часа, посчитанной по обучающим годам
    obs_lag_age_h          — сколько часов назад последнее наблюдение
                             (сеть отчитывается неравномерно)

Запуск:
    python3 scripts/postproc/add_obs_lags.py --in corpus.parquet --out corpus_lags.parquet
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

LAGS_H = [0, 6, 12, 24]
ERR_LAGS_H = [0, 6, 12, 24]
# Переменные, по которым строятся признаки: имя, столбец наблюдения, столбец
# прогноза. До 28.08.2026 признаки строились только по температуре, и регрессия
# по ветру не знала, какой ветер был перед выпуском: наблюдения давали ей
# 14,0% против 14,1%, то есть ровно ничего. Названия столбцов теперь несут имя
# переменной, иначе ветер затирал бы температуру.
VARS = [("t2m", "obs_t2m_K", "gnn_t2m"),
        ("u10", "obs_u10", "gnn_u10"),
        ("v10", "obs_v10", "gnn_v10")]


def lag_series(left, obs_s, val_col, tol, lag, index):
    """Значение val_col в момент (выпуск - lag), поиском строго назад по времени.

    Утечка из будущего невозможна по построению: direction='backward' берёт
    ближайшую запись НЕ ПОЗЖЕ искомого момента. Это свойство поиска, а не
    проверка постфактум.
    """
    q = (left.assign(_t=left["init_time_utc"] - pd.Timedelta(hours=lag))
             .sort_values("_t").reset_index().rename(columns={"index": "_row"}))
    m = pd.merge_asof(q, obs_s, on="_t", by="station_usaf",
                      direction="backward", tolerance=tol)
    out = pd.Series(np.nan, index=index, dtype="float64")
    out.loc[m["_row"].to_numpy()] = m[val_col].to_numpy()
    return out


def add_for_var(df, left, name, ocol, gcol, tol, clim_years):
    """Признаки одной переменной. Возвращает имена добавленных столбцов."""
    if ocol not in df.columns:
        print(f"[{name}] нет столбца {ocol} — пропускаю")
        return []
    obs = (df[["station_usaf", "valid_time_utc", ocol]].dropna(subset=[ocol])
             .drop_duplicates(["station_usaf", "valid_time_utc"])
             .sort_values(["station_usaf", "valid_time_utc"]).reset_index(drop=True))
    if len(obs) < 1000:
        print(f"[{name}] наблюдений всего {len(obs)} — пропускаю")
        return []
    print(f"[{name}] ряд наблюдений: {len(obs):,} записей")
    obs_s = (obs.rename(columns={"valid_time_utc": "_t"})
                .sort_values("_t").reset_index(drop=True))
    added = []
    for lag in LAGS_H:
        col = f"obs_{name}_lag{lag}"
        v = lag_series(left, obs_s, ocol, tol, lag, df.index)
        df[col] = (v - 273.15) if ocol.endswith("_K") else v
        added.append(col)
        print(f"  {col}: заполнено {df[col].notna().mean()*100:5.1f} %")
    df[f"obs_{name}_tend24"] = df[f"obs_{name}_lag0"] - df[f"obs_{name}_lag24"]
    added.append(f"obs_{name}_tend24")

    # Недавняя ошибка модели — главный признак режима. Само наблюдение режима не
    # показывает: оно несёт истину плюс шум, а режим (инверсия, застой,
    # адвекция) сидит в РАЗНИЦЕ между прогнозом и наблюдением. Для каждого
    # прошедшего момента берём прогноз с наименьшей заблаговременностью, то есть
    # самый свежий, какой на тот момент существовал.
    if gcol in df.columns:
        short = (df.loc[df[gcol].notna() & df[ocol].notna(),
                        ["station_usaf", "valid_time_utc", "lead_h", gcol, ocol]]
                   .sort_values(["station_usaf", "valid_time_utc", "lead_h"])
                   .drop_duplicates(["station_usaf", "valid_time_utc"], keep="first"))
        short["_err"] = short[ocol] - short[gcol]
        short_s = (short[["station_usaf", "valid_time_utc", "_err"]]
                     .rename(columns={"valid_time_utc": "_t"})
                     .sort_values("_t").reset_index(drop=True))
        for lag in ERR_LAGS_H:
            col = f"err_{name}_lag{lag}"
            df[col] = lag_series(left, short_s, "_err", tol, lag, df.index)
            added.append(col)
            print(f"  {col}: заполнено {df[col].notna().mean()*100:5.1f} %, "
                  f"разброс {df[col].std():.2f}")
        cols = [f"err_{name}_lag{l}" for l in ERR_LAGS_H]
        df[f"err_{name}_lag_mean"] = df[cols].mean(axis=1)
        added.append(f"err_{name}_lag_mean")
    else:
        print(f"  [!] нет столбца {gcol} — признаки недавней ошибки пропущены")

    # Климатическая норма станции по месяцу и часу — только по годам обучения.
    yr = df["valid_time_utc"].dt.year
    base = df[yr.isin(clim_years)]
    norm = (base.groupby([base["station_usaf"], base["init_time_utc"].dt.month,
                          base["init_time_utc"].dt.hour],
                         observed=True)[f"obs_{name}_lag0"].mean().rename("_norm"))
    key = pd.MultiIndex.from_arrays([df["station_usaf"],
                                     df["init_time_utc"].dt.month,
                                     df["init_time_utc"].dt.hour])
    col = f"obs_{name}_anom"
    df[col] = df[f"obs_{name}_lag0"].to_numpy() - norm.reindex(key).to_numpy()
    added.append(col)
    print(f"  {col}: заполнено {df[col].notna().mean()*100:5.1f} %")
    return added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs-col", default="obs_t2m_K")
    ap.add_argument("--gnn-col", default="gnn_t2m")
    ap.add_argument("--tolerance-h", type=float, default=1.5,
                    help="допуск при поиске наблюдения на нужный час")
    ap.add_argument("--vars", nargs="*", default=None,
                    help="какие переменные обрабатывать (по умолчанию все: "
                         + ", ".join(n for n, _, _ in VARS) + ")")
    ap.add_argument("--clim-years", type=int, nargs="*", default=None,
                    help="годы для климатической нормы (по умолчанию все, кроме последнего)")
    a = ap.parse_args()

    df = pd.read_parquet(a.inp)
    for c in ("station_usaf", "init_time_utc", "valid_time_utc"):
        if c not in df.columns:
            raise SystemExit(f"в корпусе нет столбца {c}")
    df["init_time_utc"] = pd.to_datetime(df["init_time_utc"])
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"])
    print(f"загружено {len(df):,} строк, станций {df.station_usaf.nunique()}")

    tol = pd.Timedelta(hours=a.tolerance_h)
    left = df[["station_usaf", "init_time_utc"]].copy()

    yr = df["valid_time_utc"].dt.year
    if a.clim_years:
        clim_years = list(a.clim_years)
    else:
        clim_years = sorted(int(y) for y in yr.unique())[:-1]
        print("  [!] годы для климатической нормы не заданы — беру все, кроме "
              "последнего. Если проверочная выборка не последний год, норма "
              "посчитается по ней же, и признак отклонения станет утечкой. "
              "Задавай --clim-years явно по обучающим годам.")
    print(f"климатическая норма по годам {clim_years}")

    # Возраст последнего наблюдения к моменту выпуска: сеть отчитывается
    # неравномерно, и «свежесть» сама по себе информативна. Считается один раз,
    # по основной переменной.
    prim = a.obs_col
    obs_p = (df[["station_usaf", "valid_time_utc", prim]].dropna(subset=[prim])
               .drop_duplicates(["station_usaf", "valid_time_utc"])
               .sort_values(["station_usaf", "valid_time_utc"]).reset_index(drop=True))
    obs_ps = (obs_p.rename(columns={"valid_time_utc": "_t"})
                   .sort_values("_t").reset_index(drop=True))
    obs_ps["_obs_time"] = obs_ps["_t"]
    q = (left.assign(_t=left["init_time_utc"]).sort_values("_t").reset_index()
             .rename(columns={"index": "_row"}))
    m = pd.merge_asof(q, obs_ps, on="_t", by="station_usaf",
                      direction="backward", tolerance=tol)
    age = pd.Series(np.nan, index=df.index, dtype="float64")
    age.loc[m["_row"].to_numpy()] = (
        (m["_t"] - m["_obs_time"]).dt.total_seconds() / 3600.0).to_numpy()
    df["obs_lag_age_h"] = age
    bad = int((age < 0).sum())
    assert bad == 0, f"{bad} наблюдений оказались позже выпуска — это утечка"
    print(f"obs_lag_age_h: медиана {age.median():.2f} ч, максимум {age.max():.2f} ч")

    added = ["obs_lag_age_h"]
    want = set(a.vars) if a.vars else None
    for name, ocol, gcol in VARS:
        if want is not None and name not in want:
            continue
        added += add_for_var(df, left, name, ocol, gcol, tol, clim_years)

    print(f"\nдобавлено признаков: {len(added)} — {', '.join(added)}")
    df.to_parquet(a.out, index=False)
    print(f"записано {a.out}: {len(df):,} строк, {len(df.columns)} столбцов")


if __name__ == "__main__":
    main()
