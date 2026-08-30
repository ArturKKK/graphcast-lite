#!/usr/bin/env python3
"""Вёрстка рукописи: markdown → страница для чтения и печати в PDF.

Оформление намеренно скупое, как в дипломе: одна гарнитура с засечками на весь
документ, никаких выделений в тексте, формулы набраны, а не показаны исходным
кодом. Правила печати совпадают с требованиями журнала — A4, поля 2 см,
кегль 12, интервал 1,5, — поэтому «Печать → Сохранить как PDF» даёт готовый
макет, а предпросмотр сразу показывает реальное число страниц.

Формулы собирает scripts/paper_math.py: внешние движки вроде MathJax запрещены
политикой безопасности страницы, а старый движок печати не понимает MathML.

Запуск: python3 scripts/paper_artifact.py
"""
import html
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from paper_math import CSS as MATH_CSS
from paper_math import tex_to_html  # noqa: E402
from paper_source import prepare  # noqa: E402

SRC = ROOT / "docs" / "paper" / "article_gip.md"
FIG = ROOT / "docs" / "paper" / "figures" / "fig_seam.svg"
OUT = ROOT / "docs" / "paper" / "artifact.html"


def build():
    md = SRC.read_text()
    md, done = prepare(md)
    for marker in done["cut"]:
        print(f"   отрезан служебный раздел: {marker}")

    blocks = []

    def stash(m):
        blocks.append(m.group(1).strip())
        return f"\n\nBLOCKFORMULA{len(blocks) - 1}\n\n"

    md = re.sub(r"\$\$(.+?)\$\$", stash, md, flags=re.S)
    md = re.sub(r"\$([^$\n]+)\$",
                lambda m: f'<span class="mi">{tex_to_html(m.group(1))}</span>', md)
    md = re.sub(r"\{\{ЗАПОЛНИТЬ:?\s*(.*?)\}\}",
                lambda m: f'<span class="gap">{html.escape(m.group(1))}</span>',
                md, flags=re.S)

    body = markdown.markdown(md, extensions=["tables", "attr_list"])

    for i, b in enumerate(blocks):
        body = body.replace(f"<p>BLOCKFORMULA{i}</p>",
                            f'<div class="mf">{tex_to_html(" ".join(b.split()))}</div>')

    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")

    if FIG.exists():
        svg = FIG.read_text()
        svg = svg[svg.index("<svg"):]
        anchor = "Прямая экспериментальная проверка бесшовности стыка"
        if anchor in body:
            end = body.index("</p>", body.index(anchor)) + 4
            body = (body[:end] + '\n<figure class="fig">' + svg +
                    '<figcaption>Рис. 2. Стык региональной вставки. '
                    '(а) прогноз приземной температуры на +24 ч; размер ячейки соответствует '
                    'шагу сетки — 0,25° внутри вставки и 0,703° снаружи. '
                    '(б) среднеквадратическая ошибка по расстоянию до границы, 100 сроков.'
                    '</figcaption></figure>\n' + body[end:])

    words = len(re.findall(r"\w+", re.sub(r"<[^>]+>", " ", body)))
    gaps = body.count('class="gap"')

    page = (TEMPLATE.replace("__MATHCSS__", MATH_CSS)
                    .replace("__BODY__", body)
                    .replace("__WORDS__", f"{words:,}".replace(",", " "))
                    .replace("__GAPS__", str(gaps)))
    OUT.write_text(page)
    print(f"[вёрстка] {OUT.name}: слов {words}, незаполненных мест {gaps}, "
          f"{OUT.stat().st_size // 1024} КБ")


TEMPLATE = """<title>Графовый прогноз Красноярска</title>
<style>
  :root {
    --paper:#fcfcfc; --ink:#111214; --ink-2:#5a5c60; --rule:#d9dade;
    --flag:#8a5200; --flag-bg:#fbf3e4;
    --serif:"Times New Roman","Liberation Serif",Georgia,serif;
    --sans:ui-sans-serif,system-ui,"Segoe UI",Roboto,Arial,sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper:#15161a; --ink:#e9eaee; --ink-2:#9ea1a8; --rule:#2c2e34;
      --flag:#d7a44e; --flag-bg:#241c0e;
    }
  }
  :root[data-theme="dark"] {
    --paper:#15161a; --ink:#e9eaee; --ink-2:#9ea1a8; --rule:#2c2e34;
    --flag:#d7a44e; --flag-bg:#241c0e;
  }
  body { margin:0; background:var(--paper); color:var(--ink);
         font-family:var(--serif); font-size:17px; line-height:1.6; }
  .bar { position:sticky; top:0; z-index:5; background:var(--paper);
         border-bottom:1px solid var(--rule); font-family:var(--sans);
         display:flex; flex-wrap:wrap; gap:2px 20px; padding:9px 20px;
         font-size:12px; color:var(--ink-2); }
  .bar b { color:var(--ink); font-weight:600; }
  .bar .t { margin-right:auto; }
  .wrap { max-width:35em; margin:0 auto; padding:26px 20px 80px; }
  h1 { font-size:1.5em; line-height:1.25; text-align:center; text-wrap:balance;
       margin:.3em 0 1em; font-weight:normal; }
  h2 { font-size:1.1em; margin:2em 0 .6em; font-weight:bold; text-wrap:balance; }
  h3 { font-size:1em; margin:1.4em 0 .4em; font-weight:bold; }
  p { margin:0 0 .8em; text-align:justify; hyphens:auto; }
  a { color:inherit; }
  /* Таблицы той же гарнитурой, что и текст: разнобой шрифтов в рукописи ни к чему */
  .tw { overflow-x:auto; margin:1em 0; }
  table { border-collapse:collapse; width:100%; font-size:.9em;
          font-variant-numeric:tabular-nums; }
  th, td { padding:5px 9px; border-bottom:1px solid var(--rule); text-align:left; }
  thead th { border-top:1px solid var(--ink); border-bottom:1px solid var(--ink);
             font-weight:bold; }
  td:not(:first-child), th:not(:first-child) { text-align:right; }
  .gap { display:inline; background:var(--flag-bg); color:var(--flag);
         border-bottom:1px dashed var(--flag); padding:0 3px;
         font-family:var(--sans); font-size:.8em; }
  .gap::before { content:"заполнить: "; }
  .fig { margin:1.4em 0; }
  .fig svg { width:100%; height:auto; background:#fff; }
  figcaption { font-size:.85em; color:var(--ink-2); margin-top:.5em; text-align:left; }
  ul, ol { padding-left:1.2em; }
  li { margin-bottom:.25em; }
  hr { border:0; border-top:1px solid var(--rule); margin:1.8em 0; }
__MATHCSS__
  @media print {
    @page { size:A4; margin:20mm; }
    .bar { display:none; }
    body { background:#fff; color:#000; font-size:12pt; line-height:1.5; }
    .wrap { max-width:none; padding:0; }
    h2, h3 { page-break-after:avoid; }
    table, figure, .mf { page-break-inside:avoid; }
    /* Журнальный набор: абзац задаётся отступом первой строки, а не пустой
       строкой между абзацами. Отбивка поверх полуторного интервала при полутора
       сотнях абзацев съедала около двух страниц. */
    p { margin:0; text-indent:1.25em; text-align:justify; }
    h2 + p, h3 + p, table + p, figure + p, .mf + p, ul + p, ol + p { text-indent:0; }
    p + table, p + figure, p + .mf, p + ul, p + ol { margin-top:.7em; }
    table + p, figure + p, .mf + p, ul + p, ol + p { margin-top:.7em; }
    h2 { margin:1.2em 0 .4em; }
    h3 { margin:1em 0 .3em; }
  }
</style>
<div class="bar">
  <span class="t">Рукопись для «Гидрометеорологических исследований и прогнозов»</span>
  <span>слов <b>__WORDS__</b></span>
  <span>не заполнено: <b>__GAPS__</b></span>
</div>
<div class="wrap">
__BODY__
</div>
"""

if __name__ == "__main__":
    build()
