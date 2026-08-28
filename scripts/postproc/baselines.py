#!/usr/bin/env python3
"""Простые базовые линии постобработки: что даёт таблица без нейросети.

Зачем. До сих пор нейронный постпроцессор сравнивался только с сырым прогнозом,
поэтому неизвестно, что он даёт сверх обычной таблицы поправок. Пока это
неизвестно, любое усложнение архитектуры — стрельба вслепую.

Считаются шесть уровней, от самого грубого к самому подробному:
  raw            — без поправки;
  global         — одно число на переменную;
  station        — своё смещение у каждой станции;
  st×month       — станция и месяц;
  st×month×hour  — станция, месяц и час: классическая таблица MOS;
  st×lead        — станция и срок прогноза;
  ridge          — линейная регрессия по признакам, общая для всех станций.

Про разреженные ячейки. У таблицы «станция × месяц × час» 71·12·24 ≈ 20 тысяч
ячеек, и часть из них пуста или почти пуста. Пустую заполняем родителем
(станция×месяц → станция → общее), а малонаполненную стягиваем к родителю:
b = (n·b_ячейки + k·b_родителя) / (n + k). Без этого таблица переобучается на
единичных наблюдениях и проигрывает более грубой — то есть выводы получились бы
обратными.

Запуск:
    python3 scripts/postproc/baselines.py --corpus corpus.parquet \
        --train-years 2016 2017 2018 --test-years 2020
"""
from __future__ import annotations

import argparse
import re
import numpy as np
import pandas as pd

TARGETS = [("t2m", "gnn_t2m", "obs_t2m_K", "°C"),
           ("10u", "gnn_u10", "obs_u10", "м/с"),
           ("10v", "gnn_v10", "obs_v10", "м/с")]
RIDGE_FEATS = ["gnn_t2m", "gnn_u10", "gnn_v10", "elev", "lat", "lon",
               "sin_hour", "cos_hour", "sin_doy", "cos_doy", "lead_h"]
# Признаки из наблюдений станции, известных к моменту выпуска прогноза
# (их добавляет add_obs_lags.py). До 28.08.2026 регрессия их не брала вовсе —
# список выше писался раньше, — и потому не знала о станции ничего, кроме
# координат. Ровно эти признаки различают режимы: инверсию в антициклоне от
# адвекции, тогда как таблица «станция×месяц×час» их не различает никак.
# Отбираем по образцу имени, а не списком: add_obs_lags.py строит признаки по
# каждой переменной, и список пришлось бы править следом за ним. Образец нарочно
# узкий — под него не должны попасть сами наблюдения (obs_u10, obs_t2m_K и
# прочие), иначе цель окажется среди признаков и всё «улучшение» будет утечкой.
OBS_RE = re.compile(r"^(obs|err)_[a-z0-9]+_(lag\d+|lag_mean|tend24|anom)$")
OBS_EXTRA = ["obs_lag_age_h"]


def obs_features(df) -> list:
    return [c for c in df.columns if OBS_RE.match(c)] + \
           [c for c in OBS_EXTRA if c in df.columns]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    vt = pd.to_datetime(df["valid_time_utc"])
    df = df.copy()
    df["month"] = vt.dt.month.astype("int16")
    df["hour"] = vt.dt.hour.astype("int16")
    doy = vt.dt.dayofyear.astype("float32")
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    return df


def shrunk_table(train: pd.DataFrame, resid: str, keys: list[str],
                 parent: np.ndarray, k: float) -> pd.Series:
    """Среднее по ячейке, стянутое к родителю пропорционально наполненности."""
    tmp = train[keys].copy()
    tmp["_r"] = train[resid].to_numpy()
    tmp["_p"] = parent
    g = tmp.groupby(keys, observed=True).agg(n=("_r", "size"), s=("_r", "sum"),
                                             p=("_p", "mean"))
    return (g["s"] + k * g["p"]) / (g["n"] + k)


def apply_table(df: pd.DataFrame, table: pd.Series, keys: list[str],
                fallback: np.ndarray) -> np.ndarray:
    idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
    vals = table.reindex(idx).to_numpy()
    return np.where(np.isnan(vals), fallback, vals)


def metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    d = pred - obs
    return {"rmse": float(np.sqrt(np.mean(d ** 2))),
            "mae": float(np.mean(np.abs(d))),
            "bias": float(np.mean(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--train-years", type=int, nargs="+", default=[2016, 2017, 2018])
    ap.add_argument("--test-years", type=int, nargs="+", default=[2020])
    ap.add_argument("--shrink", type=float, default=10.0,
                    help="сила стягивания редких ячеек к родителю")
    ap.add_argument("--per-lead", action="store_true", help="таблица ещё и по срокам")
    a = ap.parse_args()

    df = add_time_features(pd.read_parquet(a.corpus))
    year = pd.to_datetime(df["valid_time_utc"]).dt.year
    tr_all, te_all = df[year.isin(a.train_years)], df[year.isin(a.test_years)]
    tr, te = tr_all, te_all
    print(f"обучение {len(tr):,} строк ({a.train_years}), "
          f"проверка {len(te):,} строк ({a.test_years}), "
          f"станций {df.station_usaf.nunique()}\n")
    if len(tr) == 0 or len(te) == 0:
        raise SystemExit("пустая выборка — проверь годы")

    # Сырая ошибка по годам. Обучение сети хронологическое, последние 20%
    # отданы под контроль, то есть примерно до октября 2018 годы «выученные».
    # Если сеть их запомнила, ошибка там обязана быть заметно меньше. Это
    # проверяется прямо, а не предполагается.
    v = df[["gnn_t2m", "obs_t2m_K"]].to_numpy(np.float64)
    ok = np.isfinite(v).all(1)
    print("сырая ошибка приземной температуры по годам, °C:")
    for y, idx in df[ok].groupby(year[ok]).groups.items():
        d = df.loc[idx, "gnn_t2m"].to_numpy() - df.loc[idx, "obs_t2m_K"].to_numpy()
        print(f"    {y}: RMSE {np.sqrt((d ** 2).mean()):5.3f}  "
              f"смещение {d.mean():+5.3f}  строк {len(idx):,}")
    print()

    for name, gcol, ocol, unit in TARGETS:
        if gcol not in df.columns or ocol not in df.columns:
            print(f"[{name}] нет столбцов {gcol}/{ocol} — пропускаю\n")
            continue
        # У ветра направление в ISD-Lite сплошь и рядом отсутствует, а из него
        # считаются obs_u10/obs_v10. Строки без наблюдения надо выбрасывать для
        # каждой цели отдельно: без этого 28.08.2026 весь ветер вышел NaN, и
        # таблица показывала «nan» во всех шести строках.
        def finite(d):
            return d[np.isfinite(d[gcol].to_numpy(np.float64))
                     & np.isfinite(d[ocol].to_numpy(np.float64))]
        tr, te = finite(tr_all), finite(te_all)
        drop_tr, drop_te = len(tr_all) - len(tr), len(te_all) - len(te)
        if drop_tr or drop_te:
            print(f"[{name}] без наблюдения: обучение {drop_tr:,}, "
                  f"проверка {drop_te:,} строк — выброшены")
        if len(tr) < 1000 or len(te) < 1000:
            print(f"[{name}] наблюдений почти нет — пропускаю\n")
            continue

        tr_r = (tr[ocol] - tr[gcol]).to_numpy()          # что надо прибавить
        g_bias = float(tr_r.mean())
        tr2 = tr.assign(_resid=tr_r)

        # уровни таблицы, каждый стягивается к предыдущему
        t_st = shrunk_table(tr2, "_resid", ["station_usaf"],
                            np.full(len(tr2), g_bias), a.shrink)
        par_sm = apply_table(tr2, t_st, ["station_usaf"], np.full(len(tr2), g_bias))
        t_sm = shrunk_table(tr2, "_resid", ["station_usaf", "month"], par_sm, a.shrink)
        par_smh = apply_table(tr2, t_sm, ["station_usaf", "month"], par_sm)
        t_smh = shrunk_table(tr2, "_resid", ["station_usaf", "month", "hour"],
                             par_smh, a.shrink)
        t_sl = shrunk_table(tr2, "_resid", ["station_usaf", "lead_h"], par_sm, a.shrink)

        base_te = np.full(len(te), g_bias)
        c_st = apply_table(te, t_st, ["station_usaf"], base_te)
        c_sm = apply_table(te, t_sm, ["station_usaf", "month"], c_st)
        c_smh = apply_table(te, t_smh, ["station_usaf", "month", "hour"], c_sm)
        c_sl = apply_table(te, t_sl, ["station_usaf", "lead_h"], c_st)

        obs = te[ocol].to_numpy()
        gnn = te[gcol].to_numpy()
        rows = [("сырой прогноз", metrics(gnn, obs)),
                ("общее смещение", metrics(gnn + g_bias, obs)),
                ("станция", metrics(gnn + c_st, obs)),
                ("станция×месяц", metrics(gnn + c_sm, obs)),
                ("станция×месяц×час", metrics(gnn + c_smh, obs))]
        if a.per_lead:
            rows.append(("станция×срок", metrics(gnn + c_sl, obs)))

        base_feats = [c for c in RIDGE_FEATS if c in df.columns]
        obs_feats = obs_features(df)
        try:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline

            def ridge(feats, label, target=None, offset=None):
                """Регрессия невязки. target/offset — для настройки поверх таблицы."""
                y = tr_r if target is None else target
                add = 0.0 if offset is None else offset
                X_tr = tr[feats].to_numpy(np.float64)
                X_te = te[feats].to_numpy(np.float64)
                # Пропуски заполняем медианой ОБУЧАЮЩЕЙ выборки: считать по
                # проверочной — значит подглядеть в неё. Признаки заполнены на
                # 99%, так что это касается сотых долей строк.
                med = np.nanmedian(X_tr, axis=0)
                med = np.where(np.isfinite(med), med, 0.0)
                X_tr = np.where(np.isfinite(X_tr), X_tr, med)
                X_te = np.where(np.isfinite(X_te), X_te, med)
                m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                m.fit(X_tr, y)
                rows.append((f"{label} ({len(feats)} призн.)",
                             metrics(gnn + add + m.predict(X_te), obs)))

            ridge(base_feats, "регрессия")
            if obs_feats:
                ridge(base_feats + obs_feats, "регрессия + наблюдения")
                # Таблица и регрессия ловят разное: таблица — постоянную
                # поправку своей станции в этот месяц и час, регрессия — режим
                # по свежим наблюдениям. Соединяем: сначала таблица, потом
                # регрессия по тому, что от невязки осталось.
                #
                # Оговорка: таблица настроена на этих же обучающих строках,
                # поэтому на обучении её остаток оптимистично мал и регрессия
                # видит его заниженным. Проверочный год от этого не страдает —
                # он не участвовал ни в том, ни в другом, — так что выигрыш
                # ниже посчитан честно, но потенциал у связки, вероятно, выше.
                c_smh_tr = apply_table(tr2, t_smh,
                                       ["station_usaf", "month", "hour"], par_smh)
                ridge(base_feats + obs_feats, "таблица + регрессия",
                      target=tr_r - c_smh_tr, offset=c_smh)
        except Exception as e:  # pragma: no cover
            print(f"[{name}] регрессия не посчиталась: {e}")

        raw = rows[0][1]["rmse"]
        print(f"=== {name}, {unit} ===")
        print(f"{'способ':>34} {'RMSE':>8} {'MAE':>8} {'смещ.':>8} {'выигрыш':>9}")
        for label, m_ in rows:
            gain = (raw - m_["rmse"]) / raw * 100
            print(f"{label:>34} {m_['rmse']:8.3f} {m_['mae']:8.3f} "
                  f"{m_['bias']:+8.3f} {gain:8.1f}%")
        print()


if __name__ == "__main__":
    main()
