"""Общие части рисования полей на регулярной сетке.

Собрано из четырёх скриптов, где одно и то же повторялось двенадцать раз:

    imshow(field.T, origin="lower", cmap="jet", vmin=..., vmax=...)

Три вещи, которые повторять опасно.

Первое — ПОРЯДОК ОСЕЙ. Поля хранятся как (долгота, широта), а рисовать надо
(широта, долгота) с началом внизу, иначе север окажется снизу. Забытое
транспонирование даёт картинку, которая выглядит как карта погоды и читается
как карта погоды, но повёрнута — и заметить это по самой картинке нельзя.

Второе — ОБЩИЕ ПРЕДЕЛЫ. Правду и прогноз рядом надо рисовать в одной шкале,
иначе разница уходит в цвет и картинки становятся несравнимы, хотя выглядят
убедительно.

Третье — СИММЕТРИЯ у карт ошибки. У расходящейся шкалы середина означает нуль;
если пределы несимметричны, нулевая линия уезжает, и знак ошибки читается
неверно.

Модуль не тянет matplotlib на импорте: расчёт пределов нужен и там, где его нет.
"""
from __future__ import annotations

import numpy as np

# Последовательная шкала для самих полей. Не "jet": она неравномерна по
# восприятию — создаёт ложные полосы там, где поле гладкое, и плохо читается
# при дальтонизме. RdYlBu_r привычна в метеорологии для температуры и
# монотонна по светлоте.
SEQ_CMAP = "RdYlBu_r"

# Расходящаяся шкала для ошибок и разностей: середина — нуль.
DIV_CMAP = "RdBu_r"


def shared_limits(*fields, percentile: float | None = None) -> tuple[float, float]:
    """Общие пределы для нескольких полей, чтобы рисовать их в одной шкале.

    ``percentile`` отсекает хвосты: одна выпавшая точка иначе задаёт всю шкалу,
    и поле выходит одноцветным. Скажем, 99 оставит от 0,5 до 99,5 процентиля.
    """
    vals = np.concatenate([np.asarray(f, dtype=np.float64).ravel() for f in fields])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    if percentile is None:
        return float(vals.min()), float(vals.max())
    lo = (100.0 - percentile) / 2.0
    return float(np.percentile(vals, lo)), float(np.percentile(vals, 100.0 - lo))


def symmetric_limit(*fields, percentile: float | None = None) -> float:
    """Предел для карты ошибки: (-L, +L), чтобы нуль был ровно в середине.

    Возвращает L. Ноль превращается в единицу: шкала (0, 0) нарисовала бы
    сплошной цвет и матplotlib пожаловался бы на вырожденные пределы.
    """
    vals = np.concatenate([np.asarray(f, dtype=np.float64).ravel() for f in fields])
    vals = np.abs(vals[np.isfinite(vals)])
    if vals.size == 0:
        return 1.0
    lim = float(vals.max() if percentile is None else np.percentile(vals, percentile))
    return lim if lim > 0 else 1.0


def to_map_orientation(field) -> np.ndarray:
    """(долгота, широта) → (широта, долгота) для рисования с началом внизу.

    Одно место, где живёт это соглашение. Раньше транспонирование стояло
    двенадцатью копиями по четырём скриптам.
    """
    a = np.asarray(field)
    if a.ndim != 2:
        raise ValueError(f"ожидалось двумерное поле, получено {a.shape}")
    return a.T


def field_panel(ax, field, *, vmin=None, vmax=None, cmap=SEQ_CMAP, title=None,
                colorbar=True, label=None):
    """Нарисовать поле на готовой оси. Возвращает изображение."""
    im = ax.imshow(to_map_orientation(field), origin="lower", cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="auto")
    if title:
        ax.set_title(title)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label=label)
    return im


def error_panel(ax, err, *, limit=None, cmap=DIV_CMAP, title=None,
                colorbar=True, label=None):
    """Карта ошибки в симметричной шкале: нуль ровно в середине."""
    lim = symmetric_limit(err) if limit is None else float(limit)
    return field_panel(ax, err, vmin=-lim, vmax=lim, cmap=cmap, title=title,
                       colorbar=colorbar, label=label)
