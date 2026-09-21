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
FIGDIR = ROOT / "docs" / "paper" / "figures"
# Рисунки вставляются после абзаца, содержащего опорную фразу. Порядок в списке
# задаёт нумерацию; вставка идёт с конца, чтобы смещения не поехали.
FIGURES = [
    # Панель с ошибкой по расстоянию до границы убрана: те же результаты
    # приведены в табл. 5, а правила журнала запрещают излагать одни и те же
    # результаты одновременно таблицей и рисунком.
    ("fig_seam_map.svg",
     "Прямая проверка бесшовности приведена на рис. 1",
     "Рис. 1. Прогноз приземной температуры на +24 ч в окрестности границы "
     "региональной вставки. Размер ячейки соответствует шагу сетки — 0,25° "
     "внутри вставки и 0,703° снаружи; штриховая линия — граница вставки.",
     "Fig. 1. 2 m temperature forecast at +24 h near the insert boundary. Cell "
     "size follows the grid spacing, 0.25° inside and 0.703° outside; dashed "
     "line — the boundary."),
]

# Правило 5: подрисуночные подписи и названия таблиц — на русском и английском.
# Ключ — точное начало русской подписи таблицы в тексте статьи.
TABLES_EN = {
    "Таблица 1.": "Table 1. RMSE of 2 m temperature (°C) by lead time and aggregate skill "
                  "score S (%) per (3). 1607 initial times; region 2501 nodes, inner "
                  "zone 45 nodes.",
    "Таблица 2.": "Table 2. Error at +24 h at the insert nodes: persistence, interpolated "
                  "global forecast, multiscale model.",
    "Таблица 3.": "Table 3. Effect of loss weighting. Region, 1607 initial times; t2m "
                  "averaged over four lead times, last column — error over the whole "
                  "graph.",
    "Таблица 4.": "Table 4. RMSE of 2 m temperature (°C) and aggregate skill score for "
                  "different ways of selecting the checkpoint.",
    "Таблица 6.": "Table 6. Comparison with GraphCast on shared nodes and times: 803 "
                  "initialisations, 2501 nodes, latitude weights. RMSE averaged over "
                  "+6…+24 h; ACC against the WeatherBench 2 1990–2019 climatology.",
    "Таблица 5.": "Table 5. RMSE of 2 m temperature (°C) by distance to the insert "
                  "boundary. Positive — inside (0.25°), negative — outside (0.703°).",
}
OUT = ROOT / "docs" / "paper" / "artifact.html"


def relocate(body):
    """Таблицы и рисунки — на отдельные страницы после текста (правила, п. 13, 14).

    «В текст рисунки не вставлять», таблицы «размещаются на отдельных страницах
    после основного текста статьи». В исходнике docs/paper/article_gip.md они
    стоят рядом с обсуждением — так рукопись читается; в печать они переезжают
    в конец. Ссылки «табл. 1», «рис. 2» в тексте остаются на месте.
    """
    moved_t, moved_f = [], []

    def take(open_tag, close_tag, sink, start=0):
        """Вырезает блоки от open_tag до close_tag включительно."""
        nonlocal body
        while True:
            i = body.find(open_tag, start)
            if i < 0:
                return
            j = body.index(close_tag, i) + len(close_tag)
            sink.append(body[i:j])
            body = body[:i] + body[j:]

    # Подпись таблицы — отдельный абзац перед самой таблицей; забираем пару целиком.
    import re as _re
    seen = _re.findall(r"<strong>(Таблица \d+\.)</strong>", body)
    uncovered = [n for n in seen if n not in TABLES_EN]
    if uncovered:
        # Иначе таблица молча осталась бы в тексте без английской подписи —
        # прямое нарушение п. 5 и 13 правил, заметное только в готовом файле.
        raise SystemExit(f"[вёрстка] нет английской подписи для {uncovered}: "
                         f"допишите TABLES_EN в {Path(__file__).name}")
    for num in sorted(TABLES_EN):
        i = body.find(f"<strong>{num}</strong>")
        if i < 0:
            raise SystemExit(f"[вёрстка] в статье нет подписи «{num}»")
        i = body.rindex("<p>", 0, i)
        j = body.index("</table></div>", i) + len("</table></div>")
        cap_en = html.escape(TABLES_EN[num])
        block = body[i:j].replace("<p>", '<p class="capru">', 1)
        moved_t.append(block + f'\n<p class="capen">{cap_en}</p>')
        body = body[:i] + body[j:]

    take('<figure class="fig">', "</figure>", moved_f)

    if not moved_f:
        raise SystemExit("[вёрстка] ни один рисунок не перенесён — вставка сорвалась?")

    tail = "".join(f'\n<div class="sheet">{b}</div>\n' for b in moved_t + moved_f)
    # «На отдельных страницах после основного текста» — хвост начинается с новой.
    return body + '\n<div class="tail">' + tail + "</div>\n"


def build(standalone=None):
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

    for name, anchor, caption, caption_en in reversed(FIGURES):
        f = FIGDIR / name
        if not f.exists():
            print(f"   рисунка нет, пропускаю: {name}")
            continue
        if anchor not in body:
            # Молча пропустить нельзя: рисунок исчезает из рукописи, а сборка
            # завершается успешно. Так уже терялся рис. 2 после правки текста.
            raise SystemExit(
                f"[вёрстка] опорной фразы для {name} нет в тексте:\n"
                f"    {anchor!r}\n"
                f"    поправьте FIGURES в {Path(__file__).name} или верните фразу в статью")
        svg = f.read_text()
        svg = svg[svg.index("<svg"):]
        end = body.index("</p>", body.index(anchor)) + 4
        body = (body[:end] + '\n<figure class="fig">' + svg +
                f'<figcaption>{caption}</figcaption>'
                f'<figcaption lang="en">{caption_en}</figcaption></figure>\n' + body[end:])

    body = relocate(body)

    words = len(re.findall(r"\w+", re.sub(r"<[^>]+>", " ", body)))
    gaps = body.count('class="gap"')

    page = (TEMPLATE.replace("__MATHCSS__", MATH_CSS)
                    .replace("__BODY__", body)
                    .replace("__WORDS__", f"{words:,}".replace(",", " "))
                    .replace("__GAPS__", str(gaps)))
    OUT.write_text(page)

    # Отдельная полная страница для печати в PDF. Сам artifact.html — фрагмент
    # без <head>, он писался под внешнюю обёртку; wkhtmltopdf это переживал
    # благодаря флагу --encoding utf-8, а weasyprint такого флага не имеет и
    # кодировку угадывает — кириллица разъезжается. Поэтому для печати
    # объявляем её явно.
    if standalone:
        Path(standalone).write_text(
            '<!doctype html>\n<html lang="ru">\n<head>\n'
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            + page.split("</style>", 1)[0] + "</style>\n</head>\n<body>\n"
            + page.split("</style>", 1)[1] + "\n</body>\n</html>\n")
        print(f"[вёрстка] отдельная страница для печати: {standalone}")
    print(f"[вёрстка] {OUT.name}: слов {words}, незаполненных мест {gaps}, "
          f"{OUT.stat().st_size // 1024} КБ")
    return page


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
  table { border-collapse:collapse; width:100%; font-size:11pt;
          font-variant-numeric:tabular-nums; }
  th, td { padding:5px 9px; border-bottom:1px solid var(--rule); text-align:left; }
  thead th { border-top:1px solid var(--ink); border-bottom:1px solid var(--ink);
             font-weight:bold; }
  td:not(:first-child), th:not(:first-child) { text-align:right; }
  .gap { display:inline; background:var(--flag-bg); color:var(--flag);
         border-bottom:1px dashed var(--flag); padding:0 3px;
         font-family:var(--sans); font-size:.8em; }
  .gap::before { content:"заполнить: "; }
  .fig { margin:1.1em 0; text-align:center; }
  /* Перенесённые в конец таблицы и рисунки: каждый блок не рвётся по страницам. */
  .tail { break-before:page; page-break-before:always; }
  .sheet { margin:0 0 .8em; }
  /* Блок в целом ломать можно — иначе четыре полосы хвоста наполовину пустые.
     Нельзя ломать связку «подпись — таблица — англ. подпись»: без этого
     подпись табл. 3 оставалась внизу полосы, а сама таблица уезжала на
     следующую. */
  .capru { break-inside:avoid; break-after:avoid; page-break-after:avoid; }
  .capen { font-size:11pt; color:var(--ink-2); margin:.35em 0 0; text-indent:0; }
  /* 82 % ширины полосы: при вёрстке журнала рисунки всё равно уменьшают, а две
     полосные картинки съедали страницу сверх лимита в 20 полос. Потолок по
     высоте нужен рис. 2: карта почти квадратная, и по одной ширине она
     разворачивалась на 118 мм — полстраницы под одну панель. */
  .fig svg { width:82%; height:auto; max-height:62mm; background:#fff; }
  .fig figcaption { margin-top:.4em; }
  figcaption { font-size:11pt; color:var(--ink-2); margin-top:.5em; text-align:left; }
  /* Отступа 1,2em маркеру не хватает: он выносится влево за пределы печатной
     области и обрезается по краю страницы. У списка выводов от «1.» оставалась
     одна точка, а в списке литературы обрезались бы и двузначные номера.
     2,4em хватает на «11.» с запасом. */
  ul, ol { padding-left:2.4em; margin-left:0; }
  /* Переносы в списках. Список литературы — 32 записи с длинными
     англоязычными заглавиями; без переносов строки рвутся рано и список
     занимает на полосу больше. Содержания это не трогает. */
  li { margin-bottom:.25em; padding-left:.15em; hyphens:auto; }
  hr { border:0; border-top:1px solid var(--rule); margin:1.8em 0; }
__MATHCSS__
  @media print {
    /* Поля по правилам журнала: низ, верх и левое 25 мм, правое 15 мм. */
    @page { size:A4; margin:25mm 15mm 25mm 25mm; }
    .bar { display:none; }
    body { background:#fff; color:#000; font-size:12pt; line-height:1.5; }
    .wrap { max-width:none; padding:0; }
    h2, h3 { page-break-after:avoid; }
    table, figure, .mf { page-break-inside:avoid; }
    /* Журнальный набор: абзац задаётся отступом первой строки, а не пустой
       строкой между абзацами. Отбивка поверх полуторного интервала при полутора
       сотнях абзацев съедала около двух страниц. */
    p { margin:0; text-indent:1cm; text-align:justify; }
    /* Плотнее строка таблицы: на 30 строк пяти таблиц это целая полоса. */
    th, td { padding:2px 8px; }
    h2 + p, h3 + p, table + p, figure + p, .mf + p, ul + p, ol + p { text-indent:0; }
    p + table, p + figure, p + .mf, p + ul, p + ol { margin-top:.7em; }
    /* Список литературы набирается без отбивки между записями — при 32
       записях она съедала полосу. */
    li { margin-bottom:0; }
    table + p, figure + p, .mf + p, ul + p, ol + p { margin-top:.7em; }
    /* Отбивка заголовков: при двух десятках подпунктов четверть em на каждом
       складывается в полосу. Меньше делать нельзя — заголовок сольётся с
       предыдущим абзацем. */
    h2 { margin:.9em 0 .3em; }
    h3 { margin:.8em 0 .25em; }
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
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else None)
