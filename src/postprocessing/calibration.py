"""Мера того, можно ли верить заявленному разбросу.

Вероятностная настройка выдаёт не только поправку, но и σ — свою оценку
уверенности. Точность от этого не страдает, то есть оценка надёжности достаётся
даром. Но даром она достаётся только если ей можно верить: модель, систематически
занижающая σ, будет уверенно ошибаться, а завышающая бесполезна — «не знаю» она
скажет всегда.

Один numpy, поэтому покрыто тестами и проверяется там, где torch не установлен.
"""
from __future__ import annotations

import math

import numpy as np


def crps_gaussian(mu, sigma, obs):
    """Непрерывная оценка вероятностного прогноза. Меньше — лучше.

    Точечный прогноз — предельный случай при σ→0, и тогда оценка равна модулю
    ошибки. Значит переменная σ обязана давать меньше постоянной, иначе она не
    нужна вовсе, сколь угодно откалиброванная.

    Оценка наказывает и за промах, и за ложную уверенность, и за напрасную
    расплывчатость — тем и хороша.
    """
    mu = np.asarray(mu, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-9)
    z = (obs - mu) / sigma
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def reliability(sigma, err, n_bins: int = 10) -> list[dict]:
    """Заявленное против действительного по корзинам равного размера.

    Средние величины могут сойтись случайно — при завышенной σ на лёгких случаях
    и заниженной на трудных. Разбивка по корзинам это и ловит: у честной модели
    отношение действительной ошибки к заявленной около единицы В КАЖДОЙ корзине,
    а не только в целом.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    order = np.argsort(sigma)
    rows = []
    for part in np.array_split(order, n_bins):
        if part.size == 0:
            continue
        rows.append({"n": int(part.size),
                     "sigma_mean": float(sigma[part].mean()),
                     "rmse": float(np.sqrt((err[part] ** 2).mean()))})
    return rows


def coverage(sigma, err, k: float = 1.0) -> float:
    """Доля случаев внутри ±k·σ, в процентах.

    У правильно откалиброванной модели это 68,3 % при k=1 и 95,4 % при k=2.
    Отклонение показывает, где именно врёт оценка: в середине или в хвостах.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    return float((np.abs(err) <= k * sigma).mean() * 100.0)
