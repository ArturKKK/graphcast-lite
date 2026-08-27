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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs-col", default="obs_t2m_K")
    ap.add_argument("--gnn-col", default="gnn_t2m")
    ap.add_argument("--tolerance-h", type=float, default=1.5,
                    help="допуск при поиске наблюдения на нужный час")
    ap.add_argument("--clim-years", type=int, nargs="*", default=None,
                    help="годы для климатической нормы (по умолчанию все, кроме последнего)")
    a = ap.parse_args()

    df = pd.read_parquet(a.inp)
    for c in ("station_usaf", "init_time_utc", "valid_time_utc", a.obs_col):
        if c not in df.columns:
            raise SystemExit(f"в корпусе нет столбца {c}")
    df["init_time_utc"] = pd.to_datetime(df["init_time_utc"])
    df["valid_time_utc"] = pd.to_datetime(df["valid_time_utc"])
    print(f"загружено {len(df):,} строк, станций {df.station_usaf.nunique()}")

    # Ряд наблюдений станции: одна запись на (станция, срок действия).
    obs = (df[["station_usaf", "valid_time_utc", a.obs_col]]
           .dropna(subset=[a.obs_col])
           .drop_duplicates(["station_usaf", "valid_time_utc"])
           .sort_values(["station_usaf", "valid_time_utc"])
           .reset_index(drop=True))
    print(f"ряд наблюдений: {len(obs):,} записей")

    tol = pd.Timedelta(hours=a.tolerance_h)
    # merge_asof требует глобальной сортировки по ключу времени, а не только
    # внутри станции.
    obs_s = (obs.rename(columns={"valid_time_utc": "_t"})
                .sort_values("_t").reset_index(drop=True))
    left = df[["station_usaf", "init_time_utc"]].copy()

    # Поиск СТРОГО НАЗАД по времени: берётся ближайшее наблюдение не позже
    # искомого момента. Утечка из будущего невозможна по построению — это
    # свойство direction="backward", а не проверка постфактум.
    for lag in LAGS_H:
        q = (left.assign(_t=left["init_time_utc"] - pd.Timedelta(hours=lag))
                 .sort_values("_t").reset_index()
                 .rename(columns={"index": "_row"}))
        m = pd.merge_asof(q, obs_s, on="_t", by="station_usaf",
                          direction="backward", tolerance=tol)
        col = f"obs_t2m_lag{lag}"
        s_ = pd.Series(np.nan, index=df.index, dtype="float64")
        s_.loc[m["_row"].to_numpy()] = m[a.obs_col].to_numpy()
        df[col] = (s_ - 273.15) if a.obs_col.endswith("_K") else s_
        print(f"  {col}: заполнено {df[col].notna().mean()*100:5.1f} %")

    # Возраст последнего наблюдения к моменту выпуска: сеть отчитывается
    # неравномерно, и «свежесть» сама по себе информативна.
    q = (left.assign(_t=left["init_time_utc"]).sort_values("_t").reset_index()
             .rename(columns={"index": "_row"}))
    obs_t = obs_s.assign(_obs_time=obs_s["_t"])
    m = pd.merge_asof(q, obs_t, on="_t", by="station_usaf",
                      direction="backward", tolerance=tol)
    age = pd.Series(np.nan, index=df.index, dtype="float64")
    age.loc[m["_row"].to_numpy()] = (
        (m["_t"] - m["_obs_time"]).dt.total_seconds() / 3600.0).to_numpy()
    df["obs_lag_age_h"] = age
    bad = int((age < 0).sum())
    assert bad == 0, f"{bad} наблюдений оказались позже выпуска — это утечка"
    print(f"  obs_lag_age_h: медиана {age.median():.2f} ч, максимум {age.max():.2f} ч")

    df["obs_t2m_tend24"] = df["obs_t2m_lag0"] - df["obs_t2m_lag24"]

    # НЕДАВНЯЯ ОШИБКА МОДЕЛИ — главный признак режима.
    #
    # Само по себе наблюдение режима не показывает: оно несёт истину плюс шум, а
    # режим (инверсия, застой, адвекция) сидит в РАЗНИЦЕ между прогнозом и
    # наблюдением. Поэтому берём пару (прогноз, наблюдение) на прошедшие сроки:
    # для каждого момента — прогноз с наименьшим доступным заблаговременностью,
    # то есть самый свежий, какой на тот момент существовал.
    if a.gnn_col in df.columns:
        short = (df.loc[df[a.gnn_col].notna(), ["station_usaf", "valid_time_utc",
                                                "lead_h", a.gnn_col, a.obs_col]]
                   .sort_values(["station_usaf", "valid_time_utc", "lead_h"])
                   .drop_duplicates(["station_usaf", "valid_time_utc"], keep="first"))
        short["_err"] = short[a.obs_col] - short[a.gnn_col]
        short_s = (short[["station_usaf", "valid_time_utc", "_err"]]
                     .rename(columns={"valid_time_utc": "_t"})
                     .sort_values("_t").reset_index(drop=True))
        for lag in ERR_LAGS_H:
            q = (left.assign(_t=left["init_time_utc"] - pd.Timedelta(hours=lag))
                     .sort_values("_t").reset_index()
                     .rename(columns={"index": "_row"}))
            m = pd.merge_asof(q, short_s, on="_t", by="station_usaf",
                              direction="backward", tolerance=tol)
            col = f"err_lag{lag}"
            e = pd.Series(np.nan, index=df.index, dtype="float64")
            e.loc[m["_row"].to_numpy()] = m["_err"].to_numpy()
            df[col] = e
            print(f"  {col}: заполнено {df[col].notna().mean()*100:5.1f} %, "
                  f"разброс {df[col].std():.2f}")
        cols = [f"err_lag{l}" for l in ERR_LAGS_H]
        df["err_lag_mean"] = df[cols].mean(axis=1)
    else:
        print(f"  [!] нет столбца {a.gnn_col} — признаки недавней ошибки пропущены")

    # Климатическая норма станции по месяцу и часу — по обучающим годам.
    yr = df["valid_time_utc"].dt.year
    if a.clim_years:
        clim_years = a.clim_years
    else:
        clim_years = sorted(int(y) for y in yr.unique())[:-1]
        print("  [!] годы для климатической нормы не заданы — беру все, кроме "
              "последнего. Если проверочная выборка не последний год, норма "
              "посчитается по ней же, и признак отклонения станет утечкой. "
              "Задавай --clim-years явно по обучающим годам.")
    base = df[yr.isin(clim_years)].copy()
    base["_m"] = base["init_time_utc"].dt.month
    base["_h"] = base["init_time_utc"].dt.hour
    norm = (base.groupby(["station_usaf", "_m", "_h"], observed=True)["obs_t2m_lag0"]
                .mean().rename("_norm"))
    key = pd.MultiIndex.from_arrays([df["station_usaf"],
                                     df["init_time_utc"].dt.month,
                                     df["init_time_utc"].dt.hour])
    df["obs_t2m_anom"] = df["obs_t2m_lag0"].to_numpy() - norm.reindex(key).to_numpy()
    print(f"климатическая норма по годам {clim_years}: "
          f"заполнено {df['obs_t2m_anom'].notna().mean()*100:.1f} %")

    new = [c for c in df.columns if c.startswith(("obs_t2m_lag", "err_lag"))
           or c in ("obs_t2m_tend24", "obs_t2m_anom", "obs_lag_age_h")]
    print(f"\nдобавлено признаков: {len(new)} — {', '.join(new)}")
    df.to_parquet(a.out, index=False)
    print(f"записано {a.out}: {len(df):,} строк, {len(df.columns)} столбцов")


if __name__ == "__main__":
    main()
