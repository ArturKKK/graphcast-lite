#!/usr/bin/env python3
"""Собирает из docs/paper/article_gip.md печатную вёрстку по требованиям
журнала «Гидрометеорологические исследования и прогнозы»: Times New Roman 12,
интервал 1.5, A4 с полями 2 см.

Открыть получившийся .html в браузере и «Печать → Сохранить как PDF».
Предпросмотр печати сразу покажет реальное число страниц — это и есть проверка
лимита в 20 страниц, куда более надёжная, чем оценка по числу слов.

Использование:
    python3 scripts/paper_to_html.py                    # → docs/paper/article_gip.html
    python3 scripts/paper_to_html.py --no-appendix      # без служебных приложений
"""
import argparse
import re
from pathlib import Path

import markdown
from paper_source import prepare

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"

CSS = """
@page { size: A4; margin: 20mm; }
body {
  font-family: "Times New Roman", "Liberation Serif", Times, serif;
  font-size: 12pt; line-height: 1.5; color: #000; background: #fff;
  max-width: 170mm; margin: 0 auto; padding: 10mm 0; text-align: justify;
}
h1 { font-size: 14pt; text-align: center; margin: 18pt 0 12pt; }
h2 { font-size: 12pt; font-weight: bold; margin: 16pt 0 8pt; text-align: left; }
h3 { font-size: 12pt; font-style: italic; font-weight: normal; margin: 12pt 0 6pt; text-align: left; }
p { margin: 0 0 6pt; text-indent: 1.25cm; }
h1 + p, h2 + p, h3 + p, blockquote p, td p, th p { text-indent: 0; }
table {
  border-collapse: collapse; width: 100%; margin: 10pt 0;
  font-size: 10pt; page-break-inside: avoid;
}
th, td { border: 1px solid #000; padding: 3pt 5pt; text-align: left; }
th { font-weight: bold; text-align: center; }
td:not(:first-child) { text-align: right; }
code { font-family: "Courier New", monospace; font-size: 10pt; }
pre { font-size: 9pt; border: 1px solid #999; padding: 6pt; overflow-x: auto; }
blockquote { margin: 6pt 0 6pt 1.25cm; font-style: italic; }
ul, ol { margin: 6pt 0 6pt 1.25cm; padding-left: 12pt; }
li { margin-bottom: 3pt; }
hr { border: none; border-top: 1px solid #999; margin: 14pt 0; }
/* Незаполненные места видно сразу — чтобы не проскочили в подачу */
.todo { background: #ffe08a; padding: 0 2px; }
@media print { .todo { background: none; border-bottom: 2px dotted #000; } }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(PAPER / "article_gip.md"))
    ap.add_argument("--out", default=str(PAPER / "article_gip.html"))
    ap.add_argument("--no-appendix", action="store_true",
                    help="выбросить перечни рисунков/таблиц и замечания авторам")
    a = ap.parse_args()

    text = Path(a.src).read_text()
    text, done = prepare(text)
    for marker in done["cut"]:
        print(f"   отрезан служебный раздел: {marker}")
    n_sup = done["superscripts"]

    n_todo = len(re.findall(r"\{\{ЗАПОЛНИТЬ", text))
    # подсветить плейсхолдеры, чтобы их было видно на печати
    text = re.sub(r"(\{\{ЗАПОЛНИТЬ[^}]*\}\})", r'<span class="todo">\1</span>', text)

    html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    doc = (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
           f'<title>Мультимасштабная графовая нейросетевая модель — рукопись</title>'
           f"<style>{CSS}</style></head><body>{html_body}</body></html>")
    out = Path(a.out)
    out.write_text(doc)

    words = len(re.findall(r"\b[\w-]+\b", text))
    print(f"→ {out}")
    print(f"   слов: {words}  ·  грубая оценка ≈{words/350:.1f} стр.")
    print(f"   незаполненных мест: {n_todo} (подсвечены жёлтым)")
    print(f"   надстрочных знаков аффиляций: {n_sup}")
    print()
    print("   Открой файл в браузере и нажми Печать → Сохранить как PDF.")
    print("   Число страниц в предпросмотре и есть проверка лимита журнала (20 стр.).")


if __name__ == "__main__":
    main()
