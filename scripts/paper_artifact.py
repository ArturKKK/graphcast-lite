#!/usr/bin/env python3
"""Собирает страницу рукописи для чтения и печати в PDF.

Отличие от paper_to_html.py: тот делает строго печатную вёрстку по метрикам
журнала, а этот — читаемую страницу с оглавлением, подсветкой незаполненных
мест и вшитым рисунком, которую удобно смотреть с телефона и из которой всё так
же печатается PDF (правила @page те же).

Формулы: движка вроде MathJax в артефакте не будет — политика безопасности
режет внешние скрипты. Поэтому строчная математика переводится в юникод, а
блочные формулы показываются отдельными плашками с исходной записью.

Запуск: python3 scripts/paper_artifact.py
"""
import html
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "paper" / "article_gip.md"
FIG = ROOT / "docs" / "paper" / "figures" / "fig_seam.svg"
OUT = ROOT / "docs" / "paper" / "artifact.html"

GREEK = {
    r"\sigma": "σ", r"\tau": "τ", r"\alpha": "α", r"\varphi": "φ", r"\phi": "φ",
    r"\mu": "μ", r"\lambda": "λ", r"\Delta": "Δ", r"\delta": "δ", r"\pi": "π",
    r"\in": "∈", r"\cdot": "·", r"\times": "×", r"\approx": "≈", r"\le": "≤",
    r"\ge": "≥", r"\sum": "Σ", r"\qquad": "  ", r"\,": " ", r"\;": " ",
}


def inline_math(m):
    """Строчная формула → юникод, насколько это возможно."""
    t = m.group(1)
    t = re.sub(r"\\bar\{([^}]*)\}", lambda g: g.group(1) + "\u0304", t)
    t = re.sub(r"\\overline\{([^}]*)\}", lambda g: g.group(1) + "\u0304", t)
    t = re.sub(r"\\hat\{([^}]*)\}", lambda g: g.group(1) + "\u0302", t)
    t = re.sub(r"\\(mathbf|mathrm|mathcal|text)\{([^}]*)\}", r"\2", t)
    t = t.replace("{,}", ",")
    for k, v in GREEK.items():
        t = t.replace(k, v)
    t = re.sub(r"_\{([^}]*)\}", r"\1", t).replace("_", "")
    t = t.replace("{", "").replace("}", "").replace("\\", "")
    return f'<span class="m">{html.escape(t)}</span>'


def build():
    md = SRC.read_text()

    # Блочные формулы — вынести до разметки, чтобы markdown их не трогал
    blocks = []

    def stash(m):
        blocks.append(m.group(1).strip())
        return f"\n\nBLOCKFORMULA{len(blocks) - 1}\n\n"

    md = re.sub(r"\$\$(.+?)\$\$", stash, md, flags=re.S)
    md = re.sub(r"\$([^$\n]+)\$", inline_math, md)

    # Незаполненные места
    md = re.sub(r"\{\{ЗАПОЛНИТЬ:?\s*(.*?)\}\}",
                lambda m: f'<span class="gap"><b>не заполнено:</b> {html.escape(m.group(1))}</span>',
                md, flags=re.S)

    body = markdown.markdown(md, extensions=["tables", "attr_list"])

    for i, b in enumerate(blocks):
        body = body.replace(
            f"<p>BLOCKFORMULA{i}</p>",
            f'<div class="formula"><code>{html.escape(b)}</code></div>')

    # Таблицы — в прокручиваемые контейнеры
    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")

    # Рисунок стыка
    if FIG.exists():
        svg = FIG.read_text()
        svg = svg[svg.index("<svg"):]
        # Фраза стоит в середине абзаца, поэтому ищем её саму и вставляем
        # рисунок после конца объемлющего абзаца.
        anchor = "Прямая экспериментальная проверка бесшовности стыка"
        if anchor in body:
            end = body.index("</p>", body.index(anchor)) + 4
            body = (body[:end] + '\n<figure class="fig">' + svg +
                    '<figcaption><b>Рис. 2.</b> Стык региональной вставки. '
                    '(а) прогноз приземной температуры на +24 ч; размер ячейки соответствует '
                    'шагу сетки — 0,25° внутри вставки и 0,703° снаружи. '
                    '(б) среднеквадратическая ошибка по расстоянию до границы, 100 сроков.'
                    '</figcaption></figure>\n' + body[end:])

    words = len(re.findall(r"\w+", re.sub(r"<[^>]+>", " ", body)))
    gaps = body.count('class="gap"')
    pages = round(words / 350 + 3)

    page = TEMPLATE.replace("__BODY__", body).replace("__WORDS__", f"{words:,}".replace(",", " ")) \
                   .replace("__GAPS__", str(gaps)).replace("__PAGES__", str(pages))
    OUT.write_text(page)
    print(f"[artifact] {OUT} — слов {words}, пробелов {gaps}, ≈{pages} стр., "
          f"{OUT.stat().st_size // 1024} КБ")


TEMPLATE = """<title>Графовый прогноз Красноярска</title>
<style>
  :root {
    --paper: #fcfcfd; --card: #ffffff; --ink: #16181d; --ink-2: #565b66;
    --rule: #dde0e7; --accent: #1c5cab; --flag: #a4600a; --flag-bg: #fdf5e6;
    --serif: "Times New Roman", "Liberation Serif", Georgia, serif;
    --sans: ui-sans-serif, system-ui, "Segoe UI", Roboto, Arial, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper: #14161a; --card: #191c22; --ink: #e8eaf0; --ink-2: #a0a6b3;
      --rule: #2a2e37; --accent: #6da7ec; --flag: #d9a441; --flag-bg: #241d10;
    }
  }
  :root[data-theme="dark"] {
    --paper: #14161a; --card: #191c22; --ink: #e8eaf0; --ink-2: #a0a6b3;
    --rule: #2a2e37; --accent: #6da7ec; --flag: #d9a441; --flag-bg: #241d10;
  }
  body { margin: 0; background: var(--paper); color: var(--ink);
         font-family: var(--serif); font-size: 17px; line-height: 1.62; }
  .bar { position: sticky; top: 0; z-index: 5; background: var(--card);
         border-bottom: 1px solid var(--rule); font-family: var(--sans);
         display: flex; flex-wrap: wrap; gap: 4px 22px; align-items: baseline;
         padding: 10px 20px; font-size: 13px; color: var(--ink-2); }
  .bar b { color: var(--ink); font-weight: 600; }
  .bar .t { font-size: 14px; color: var(--ink); margin-right: auto; }
  .wrap { max-width: 36em; margin: 0 auto; padding: 28px 20px 80px; }
  h1 { font-size: 1.55em; line-height: 1.25; text-wrap: balance; margin: .2em 0 .6em; }
  h2 { font-size: 1.18em; margin: 2.1em 0 .5em; text-wrap: balance;
       padding-bottom: .25em; border-bottom: 1px solid var(--rule); }
  h3 { font-size: 1.02em; margin: 1.5em 0 .35em; color: var(--accent); }
  p { margin: 0 0 .85em; text-align: justify; hyphens: auto; }
  a { color: var(--accent); }
  code { font-family: ui-monospace, "SF Mono", Consolas, monospace; font-size: .88em; }
  .m { font-style: italic; white-space: nowrap; }
  .formula { background: var(--card); border: 1px solid var(--rule); border-left: 3px solid var(--accent);
             padding: 10px 14px; margin: 1.1em 0; overflow-x: auto; }
  .formula code { font-size: .82em; color: var(--ink-2); white-space: pre; }
  .gap { display: inline-block; background: var(--flag-bg); color: var(--flag);
         border: 1px dashed var(--flag); border-radius: 3px; padding: 1px 7px;
         font-family: var(--sans); font-size: .78em; line-height: 1.45; }
  .tw { overflow-x: auto; margin: 1.1em 0; border: 1px solid var(--rule); }
  table { border-collapse: collapse; width: 100%; font-family: var(--sans);
          font-size: .82em; font-variant-numeric: tabular-nums; }
  th, td { padding: 6px 10px; border-bottom: 1px solid var(--rule); text-align: left; }
  th { background: var(--card); font-weight: 600; color: var(--ink-2);
       text-transform: uppercase; letter-spacing: .04em; font-size: .92em; }
  td:not(:first-child), th:not(:first-child) { text-align: right; }
  .fig { margin: 1.6em 0; padding: 0; }
  .fig svg { width: 100%; height: auto; background: #fff; border: 1px solid var(--rule); }
  figcaption { font-family: var(--sans); font-size: .8em; color: var(--ink-2);
               margin-top: .6em; text-align: left; }
  ul, ol { padding-left: 1.3em; }
  li { margin-bottom: .3em; }
  hr { border: 0; border-top: 1px solid var(--rule); margin: 2em 0; }
  @media print {
    @page { size: A4; margin: 20mm; }
    .bar { display: none; }
    body { background: #fff; color: #000; font-size: 12pt; line-height: 1.5; }
    .wrap { max-width: none; padding: 0; }
    .fig svg { border: 0; }
  }
</style>
<div class="bar">
  <span class="t">Рукопись для «Гидрометеорологических исследований и прогнозов»</span>
  <span>слов <b>__WORDS__</b></span>
  <span>примерно <b>__PAGES__</b> стр. при лимите <b>20</b></span>
  <span>не заполнено мест: <b>__GAPS__</b></span>
</div>
<div class="wrap">
__BODY__
</div>
"""

if __name__ == "__main__":
    build()
