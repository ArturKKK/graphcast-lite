#!/usr/bin/env python3
"""Рисунок «стык вставки» для статьи: карта поля и профиль ошибки.

Панель (а) — прогноз приземной температуры на +24 ч в окрестности границы
региональной вставки. Каждый узел нарисован прямоугольником своего шага: 0,25°
внутри вставки и 0,703° снаружи. Видно, есть ли разрыв поля на склейке.

Панель (б) — среднеквадратическая ошибка в зависимости от расстояния до границы.
Положительное расстояние — внутри вставки, отрицательное — снаружи.

Рисуется векторно, без matplotlib: на машине его нет, а для печати SVG даже
удобнее — не пикселится и правится текстовым редактором.

Запуск:
  python3 scripts/paper_fig_seam.py
"""
import json
import os

import numpy as np

SRC = "docs/paper/runs/vm4_seam/seam_map_data.npz"
OUT = "docs/paper/figures/fig_seam.svg"

# Область интереса (границы вставки)
ROI = (50.0, 60.0, 83.0, 98.0)
# Окно карты — чуть шире вставки, чтобы стык был в середине
WIN = (47.0, 63.0, 79.0, 102.0)

# Последовательная шкала: один тон, светлый → тёмный (см. палитру)
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
C_INS = "#2a78d6"   # категориальный слот 1 — вставка
C_GLB = "#eb6834"   # слот 2 — глобальная сетка
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d8d4"

# Профиль из seam_profile.md, горизонт +24 ч
PROF_INS = [(12.5, 2.12, "0–25"), (37.5, 2.08, "25–50"), (75, 2.02, "50–100"),
            (150, 1.99, "100–200"), (600, 2.07, "200–1000")]
PROF_GLB = [(-50, 1.87, "−100–0"), (-200, 1.95, "−300–−100"), (-650, 2.05, "−1000–−300")]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    z = np.load(SRC)
    lat, lon = z["lat"], z["lon"]
    isreg = z["is_regional"]
    val = z["pred_last"]
    var = str(z["variable"])
    n = int(z["n_samples"])

    m = ((lat >= WIN[0]) & (lat <= WIN[1]) & (lon >= WIN[2]) & (lon <= WIN[3]))
    lat, lon, isreg, val = lat[m], lon[m], isreg[m], val[m]
    # Поле в градусах Цельсия, если пришло в кельвинах
    if np.nanmedian(val) > 150:
        val = val - 273.15
    lo, hi = np.nanpercentile(val, [2, 98])
    print(f"[fig] узлов в окне {len(lat)} (вставка {int(isreg.sum())}), "
          f"{var}: {lo:.1f}…{hi:.1f} °C, сроков {n}")

    # --- геометрия панелей ---
    W, H = 1020, 440
    ax, ay, aw, ah = 56, 40, 430, 340          # панель (а)
    bx, by, bw, bh = 610, 40, 370, 340         # панель (б)

    def mx(v):   # долгота → x
        return ax + (v - WIN[2]) / (WIN[3] - WIN[2]) * aw

    def my(v):   # широта → y (вверх)
        return ay + ah - (v - WIN[0]) / (WIN[1] - WIN[0]) * ah

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'width="{W}" height="{H}" font-family="Times New Roman, Times, serif">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>']

    # --- панель (а): узлы ---
    cellw_g = (WIN[3] - WIN[2]) / aw          # градусов на пиксель, долгота
    cellh_g = (WIN[1] - WIN[0]) / ah
    for i in range(len(lat)):
        step = 0.25 if isreg[i] else 0.703
        w = max(step / cellw_g, 1.2)
        h = max(step / cellh_g, 1.2)
        t = (val[i] - lo) / (hi - lo)
        t = 0.0 if not np.isfinite(t) else min(1.0, max(0.0, t))
        col = RAMP[min(len(RAMP) - 1, int(t * len(RAMP)))]
        s.append(f'<rect x="{mx(lon[i]) - w/2:.1f}" y="{my(lat[i]) - h/2:.1f}" '
                 f'width="{w:.1f}" height="{h:.1f}" fill="{col}"/>')

    # граница вставки
    s.append(f'<rect x="{mx(ROI[2]):.1f}" y="{my(ROI[1]):.1f}" '
             f'width="{mx(ROI[3]) - mx(ROI[2]):.1f}" height="{my(ROI[0]) - my(ROI[1]):.1f}" '
             f'fill="none" stroke="{INK}" stroke-width="1.6" stroke-dasharray="6 3"/>')
    s.append(f'<rect x="{ax}" y="{ay}" width="{aw}" height="{ah}" fill="none" '
             f'stroke="{INK2}" stroke-width="1"/>')

    # подписи осей карты
    for v in (80, 85, 90, 95, 100):
        s.append(f'<text x="{mx(v):.1f}" y="{ay + ah + 16}" font-size="12" fill="{INK2}" '
                 f'text-anchor="middle">{v}°в.д.</text>')
    for v in (50, 55, 60):
        s.append(f'<text x="{ax - 6}" y="{my(v) + 4:.1f}" font-size="12" fill="{INK2}" '
                 f'text-anchor="end">{v}°с.ш.</text>')
    s.append(f'<text x="{ax}" y="{ay - 14}" font-size="14" fill="{INK}">'
             f'(а) прогноз приземной температуры на +24 ч</text>')

    # шкала цвета
    lx, ly, lw, lh = ax, ay + ah + 30, 200, 10
    for k, c in enumerate(RAMP):
        s.append(f'<rect x="{lx + k * lw / len(RAMP):.1f}" y="{ly}" '
                 f'width="{lw / len(RAMP) + 0.5:.1f}" height="{lh}" fill="{c}"/>')
    s.append(f'<rect x="{lx}" y="{ly}" width="{lw}" height="{lh}" fill="none" '
             f'stroke="{INK2}" stroke-width="0.8"/>')
    s.append(f'<text x="{lx}" y="{ly + lh + 14}" font-size="11" fill="{INK2}">{lo:.0f} °C</text>')
    s.append(f'<text x="{lx + lw}" y="{ly + lh + 14}" font-size="11" fill="{INK2}" '
             f'text-anchor="end">{hi:.0f} °C</text>')
    s.append(f'<text x="{lx + lw + 16}" y="{ly + lh - 1}" font-size="11" fill="{INK2}">'
             f'штриховая линия — граница вставки 0,25°</text>')

    # --- панель (б): профиль ---
    xs = [p[0] for p in PROF_GLB + PROF_INS]
    ys = [p[1] for p in PROF_GLB + PROF_INS]
    x0, x1 = -1100, 1100
    y0, y1 = 1.75, 2.25

    def px(v):
        return bx + (v - x0) / (x1 - x0) * bw

    def py(v):
        return by + bh - (v - y0) / (y1 - y0) * bh

    for gv in (1.8, 1.9, 2.0, 2.1, 2.2):
        s.append(f'<line x1="{bx}" y1="{py(gv):.1f}" x2="{bx + bw}" y2="{py(gv):.1f}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{bx - 6}" y="{py(gv) + 4:.1f}" font-size="12" fill="{INK2}" '
                 f'text-anchor="end">{gv:.1f}</text>')
    # ось «граница»
    s.append(f'<line x1="{px(0):.1f}" y1="{by}" x2="{px(0):.1f}" y2="{by + bh}" '
             f'stroke="{INK}" stroke-width="1.4" stroke-dasharray="6 3"/>')
    s.append(f'<text x="{px(0):.1f}" y="{by - 6}" font-size="12" fill="{INK}" '
             f'text-anchor="middle">граница</text>')

    def poly(pts, col, marker):
        d = " ".join(f"{px(a):.1f},{py(b):.1f}" for a, b, _ in pts)
        s.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        for a, b, _ in pts:
            if marker == "circle":
                s.append(f'<circle cx="{px(a):.1f}" cy="{py(b):.1f}" r="4.5" fill="{col}" '
                         f'stroke="#ffffff" stroke-width="2"/>')
            else:
                s.append(f'<rect x="{px(a) - 4:.1f}" y="{py(b) - 4:.1f}" width="8" height="8" '
                         f'fill="{col}" stroke="#ffffff" stroke-width="2"/>')

    poly(PROF_GLB, C_GLB, "square")
    poly(PROF_INS, C_INS, "circle")
    s.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="none" '
             f'stroke="{INK2}" stroke-width="1"/>')
    for v in (-1000, -500, 0, 500, 1000):
        s.append(f'<text x="{px(v):.1f}" y="{by + bh + 16}" font-size="12" fill="{INK2}" '
                 f'text-anchor="middle">{v}</text>')
    s.append(f'<text x="{bx + bw/2:.1f}" y="{by + bh + 34}" font-size="12" fill="{INK2}" '
             f'text-anchor="middle">расстояние до границы, км</text>')
    s.append(f'<text x="{bx}" y="{by - 24}" font-size="14" fill="{INK}">'
             f'(б) ошибка на +24 ч по расстоянию до границы, °C</text>')

    # легенда: две серии, обе подписаны прямо
    s.append(f'<circle cx="{bx + 14}" cy="{by + bh + 56}" r="4.5" fill="{C_INS}"/>')
    s.append(f'<text x="{bx + 26}" y="{by + bh + 60}" font-size="12" fill="{INK2}">'
             f'внутри вставки, 0,25°</text>')
    s.append(f'<rect x="{bx + 190}" y="{by + bh + 52}" width="8" height="8" fill="{C_GLB}"/>')
    s.append(f'<text x="{bx + 204}" y="{by + bh + 60}" font-size="12" fill="{INK2}">'
             f'глобальная сетка, 0,703°</text>')

    s.append('</svg>')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(s))
    print(f"[fig] сохранено → {OUT} ({os.path.getsize(OUT) // 1024} КБ)")


if __name__ == "__main__":
    main()
