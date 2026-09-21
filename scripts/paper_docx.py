#!/usr/bin/env python3
"""Сборка рукописи в Word по требованиям журнала.

Зачем. «Гидрометеорологические исследования и прогнозы» принимают только Word:
«Текст набирается в формате Word шрифтом Times New Roman 12 кеглем на листе
форматом А4 с полями: нижнее, верхнее и левое – 25 мм, правое – 15 мм.
Выравнивание по ширине. Абзацный отступ 1 см» (Правила для авторов, п. 6).
PDF, который мы собираем weasyprint, нужен нам для вычитки и счёта полос, а в
редакцию идёт docx.

Формулы. Правила требуют «редакторы формул Microsoft MathType или Equation
Editor» (п. 15). pandoc переводит $…$ в OMML — это и есть родной формат
Equation Editor, то есть формулы приходят редактируемыми, а не картинками.

Таблицы и рисунки, как и в PDF, уезжают на отдельные страницы после текста
(п. 13, 14). Правило разметки то же самое, что в paper_artifact.py, но
выполняется прямо над разметкой: docx собирается pandoc-ом, а не из HTML.

Запуск:
    python3 scripts/paper_docx.py
    python3 scripts/paper_docx.py --out /tmp/статья.docx
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from paper_artifact import FIGURES, TABLES_EN  # noqa: E402
from paper_source import cut_service_sections  # noqa: E402

SRC = ROOT / "docs" / "paper" / "article_gip.md"
FIGDIR = ROOT / "docs" / "paper" / "figures" / "submission"
OUT = ROOT / "docs" / "paper" / "article_gip.docx"

# Твипы: 1 мм = 56,7. Поля 25/15 мм, лист А4 210×297 мм.
MM = 56.7
PG = dict(w=round(210 * MM), h=round(297 * MM))
MAR = dict(top=round(25 * MM), bottom=round(25 * MM),
           left=round(25 * MM), right=round(15 * MM))
LINE = 360      # полуторный интервал: 1,5 × 240
INDENT = 567    # абзацный отступ 1 см
FONT = "Times New Roman"


def reference_docx(dst: Path) -> Path:
    """Эталонный документ pandoc, поправленный под требования журнала."""
    base = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                          capture_output=True, check=True).stdout
    tmp = dst.with_suffix(".base.docx")
    tmp.write_bytes(base)

    with zipfile.ZipFile(tmp) as z:
        parts = {n: z.read(n) for n in z.namelist()}

    st = parts["word/styles.xml"].decode("utf-8")
    # Гарнитура и кегль всему документу сразу: правила требуют TNR 12 и не
    # допускают разнобоя, а pandoc по умолчанию ставит тему Calibri.
    st = st.replace(
        '<w:rFonts w:asciiTheme="minorHAnsi" w:eastAsiaTheme="minorHAnsi" '
        'w:hAnsiTheme="minorHAnsi" w:cstheme="minorBidi" />',
        f'<w:rFonts w:ascii="{FONT}" w:eastAsia="{FONT}" '
        f'w:hAnsi="{FONT}" w:cs="{FONT}" />')
    # Полуторный интервал, выравнивание по ширине, отступ первой строки.
    # Отбивку после абзаца убираем: абзац в журнальном наборе задаётся
    # отступом, а не пустой строкой.
    st = st.replace(
        '<w:pPr>\n        <w:spacing w:after="200" />\n      </w:pPr>',
        f'<w:pPr><w:spacing w:after="0" w:line="{LINE}" w:lineRule="auto" />'
        f'<w:jc w:val="both" /><w:ind w:firstLine="{INDENT}" /></w:pPr>')

    # Заголовки: тем же кеглем, жирным, без отступа первой строки (п. 6).
    heads = "".join(
        f'<w:style w:type="paragraph" w:styleId="Heading{i}">'
        f'<w:name w:val="heading {i}" /><w:basedOn w:val="Normal" /><w:qFormat />'
        f'<w:pPr><w:keepNext /><w:spacing w:before="240" w:after="120" />'
        f'<w:jc w:val="left" /><w:ind w:firstLine="0" /></w:pPr>'
        f'<w:rPr><w:b /><w:sz w:val="24" /></w:rPr></w:style>'
        for i in (1, 2, 3))
    st = st.replace("</w:styles>", heads + "</w:styles>")
    parts["word/styles.xml"] = st.encode("utf-8")

    doc = parts["word/document.xml"].decode("utf-8")
    doc = doc.replace("<w:sectPr>", (
        "<w:sectPr>"
        f'<w:pgSz w:w="{PG["w"]}" w:h="{PG["h"]}" />'
        f'<w:pgMar w:top="{MAR["top"]}" w:right="{MAR["right"]}" '
        f'w:bottom="{MAR["bottom"]}" w:left="{MAR["left"]}" '
        'w:header="708" w:footer="708" w:gutter="0" />'), 1)
    parts["word/document.xml"] = doc.encode("utf-8")

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    tmp.unlink()
    return dst


def relocate(md: str) -> str:
    """Таблицы и рисунки — после основного текста (правила, п. 13, 14)."""
    moved = []
    for num in sorted(TABLES_EN):
        m = re.search(rf"^\*\*{re.escape(num)}\*\*.*?(?=\n\n[^|])", md,
                      flags=re.S | re.M)
        if not m:
            raise SystemExit(f"[docx] не нашёл в тексте блок «{num}»")
        moved.append(m.group(0).rstrip() + "\n\n" + TABLES_EN[num])
        md = md[:m.start()] + md[m.end():]

    figs = []
    for i, (name, _, caption, caption_en) in enumerate(FIGURES, 1):
        img = FIGDIR / f"fig{i}.png"
        if not img.exists():
            raise SystemExit(
                f"[docx] нет {img} — сначала:\n"
                f"    python3 scripts/paper_figures_export.py --png")
        figs.append(f"![]({img})\n\n{caption}\n\n{caption_en}")

    md = re.sub(r"\n{3,}", "\n\n", md).rstrip()
    tail = "\n\n".join(moved + figs)
    return md + "\n\n\\newpage\n\n" + tail + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--src", default=str(SRC))
    a = ap.parse_args()

    if not shutil.which("pandoc"):
        raise SystemExit("нет pandoc: apt-get install -y pandoc")

    md, cut = cut_service_sections(Path(a.src).read_text())
    for marker in cut:
        print(f"   отрезан служебный раздел: {marker}")
    md = relocate(md)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src_md = tmp / "article.md"
        # Надстрочные ^1,2^ оставляем как есть: в разметке pandoc это и есть
        # надстрочный знак, а замена на <sup> дала бы в Word голый тег.
        src_md.write_text(md)
        ref = reference_docx(tmp / "reference.docx")
        r = subprocess.run(
            ["pandoc", str(src_md), "-f", "markdown+superscript+subscript",
             "-t", "docx", "--reference-doc", str(ref),
             "--resource-path", str(ROOT), "-o", a.out],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"[docx] pandoc не справился:\n{r.stderr[-2000:]}")

    size = Path(a.out).stat().st_size // 1024
    print(f"готово: {a.out} ({size} КБ)")
    print("проверьте в Word: TNR 12, полуторный интервал, поля 25/25/25/15 мм,")
    print("формулы редактируемые (OMML), таблицы и рисунки после текста")


if __name__ == "__main__":
    main()
