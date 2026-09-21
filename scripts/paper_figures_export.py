#!/usr/bin/env python3
"""Экспорт рисунков рукописи в отдельные растровые файлы.

Правила журнала, п. 14: «Рисунки также должны быть подготовлены в отдельных
файлах в графических форматах .jpg, .tif (для возможного их редактирования),
должны быть четкими, с учетом последующего уменьшения». В репозитории рисунки
лежат в SVG — из них и печатается PDF, — поэтому растр собирается отсюда же и
не может разойтись с тем, что в рукописи.

Путь конверсии: SVG -> PDF (weasyprint, тот же движок, что печатает статью) ->
растр (pdftoppm из poppler). Отдельного конвертера SVG в этой среде нет, а
цепочка через weasyprint заодно гарантирует, что растр выглядит ровно так же,
как рисунок в собранном PDF.

  python scripts/paper_figures_export.py            # tif 600 dpi
  python scripts/paper_figures_export.py --jpeg --dpi 300
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from paper_artifact import FIGDIR, FIGURES  # noqa: E402

OUT = ROOT / "docs" / "paper" / "figures" / "submission"


def size_mm(svg, width_mm):
    """Высота страницы под рисунок по соотношению сторон из viewBox."""
    m = re.search(r'viewBox="[\d.\s-]*?([\d.]+)\s+([\d.]+)"', svg)
    if not m:
        raise SystemExit("в SVG нет viewBox — не вычислить пропорции")
    w, h = float(m.group(1)), float(m.group(2))
    return width_mm, width_mm * h / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--jpeg", action="store_true", help="вместо .tif")
    ap.add_argument("--png", action="store_true",
                    help="вместо .tif: нужен для вставки в docx, Word не берёт SVG")
    ap.add_argument("--width-mm", type=float, default=160.0,
                    help="ширина рисунка в журнале, мм")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for i, (name, _, _, _) in enumerate(FIGURES, 1):
        svg = (FIGDIR / name).read_text()
        svg = svg[svg.index("<svg"):]
        w, h = size_mm(svg, a.width_mm)
        page = (f'<!doctype html><html><head><meta charset="utf-8"><style>'
                f'@page {{ size:{w:.2f}mm {h:.2f}mm; margin:0 }}'
                f'html,body {{ margin:0; padding:0; background:#fff }}'
                f'svg {{ display:block; width:{w:.2f}mm; height:{h:.2f}mm }}'
                f'</style></head><body>{svg}</body></html>')
        with tempfile.TemporaryDirectory() as tmp:
            html = Path(tmp) / "f.html"
            pdf = Path(tmp) / "f.pdf"
            html.write_text(page)
            subprocess.run([sys.executable, "-m", "weasyprint", str(html), str(pdf)],
                           check=True, capture_output=True)
            stem = OUT / f"fig{i}"
            # Без сжатия .tif 600 dpi весит десятки мегабайт — рукопись подаётся
            # по электронной почте, поэтому LZW (без потерь) обязателен.
            if a.png:
                fmt = ["-png"]
            elif a.jpeg:
                fmt = ["-jpeg", "-jpegopt", "quality=95"]
            else:
                fmt = ["-tiff", "-tiffcompression", "lzw"]
            subprocess.run(["pdftoppm", *fmt, "-r", str(a.dpi), "-singlefile",
                            str(pdf), str(stem)], check=True)
        # Именно ожидаемое имя, а не glob: рядом лежит файл прошлого прогона в
        # другом формате, и glob отчитывался чужим файлом.
        ext = "png" if a.png else ("jpg" if a.jpeg else "tif")
        got = OUT / f"fig{i}.{ext}"
        if not got.exists():
            raise SystemExit(f"pdftoppm не создал {got}")
        made.append(got)
        print(f"  рис. {i}: {got.name}  {w:.0f}×{h:.0f} мм, {a.dpi} dpi, "
              f"{got.stat().st_size // 1024} КБ")

    if len(made) != len(FIGURES):
        raise SystemExit("собрались не все рисунки")
    print(f"готово: {OUT}")


if __name__ == "__main__":
    main()
