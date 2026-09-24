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

Номера формул. pandoc при переводе в OMML молча выбрасывает \\tag: до
24.09.2026 в docx не было номеров (1)–(7), хотя текст на них ссылается.
Номера снимаются из разметки до pandoc, а после него каждая выключная формула
ставится в таблицу без рамок: формула по центру, номер у правого поля. Так
номера оформляют в Word, и формула остаётся редактируемой.

Формат .doc (--doc). Яндекс Документы при загрузке docx теряют формулы OMML.
В .doc LibreOffice сохраняет их объектами Microsoft Equation 3.0, у каждого
есть готовое изображение. Это запасной путь для совместной правки, в редакцию
идёт docx.

Таблицы и рисунки, как и в PDF, уезжают на отдельные страницы после текста
(п. 13, 14). Правило разметки то же самое, что в paper_artifact.py, но
выполняется прямо над разметкой: docx собирается pandoc-ом, а не из HTML.

Запуск:
    python3 scripts/paper_docx.py
    python3 scripts/paper_docx.py --out /tmp/статья.docx
    python3 scripts/paper_docx.py --doc      # ещё и article_gip.doc рядом
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


TAG = re.compile(r"\\tag\{([^}]*)\}")
TEXT_W = PG["w"] - MAR["left"] - MAR["right"]    # ширина полосы набора, твипы
NUM_W = 850                                        # колонка номера, 1,5 см


def strip_tags(md: str):
    """Снимает \\tag{n} с выключных формул; возвращает разметку и номера по порядку."""
    tags = []

    def one(m):
        body = m.group(1)
        t = TAG.search(body)
        tags.append(t.group(1) if t else None)
        # в одну строку: после снятия \\tag остаётся пустая строка, и pandoc
        # перестаёт считать формулу выключной
        return "$$" + " ".join(TAG.sub("", body).split()) + "$$"

    return re.sub(r"\$\$(.+?)\$\$", one, md, flags=re.S), tags


def number_equations(docx: Path, tags: list) -> int:
    """Каждый абзац с выключной формулой → таблица «формула | (n)» без рамок."""
    with zipfile.ZipFile(docx) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    doc = parts["word/document.xml"].decode("utf-8")
    paras = list(re.finditer(r"<w:p>(?:(?!</w:p>).)*?<m:oMathPara>.*?</m:oMathPara>.*?</w:p>",
                             doc, flags=re.S))
    if len(paras) != len(tags):
        raise SystemExit(f"[docx] выключных формул в docx {len(paras)}, в разметке {len(tags)}")
    none = ('<w:tblBorders>' + "".join(f'<w:{s} w:val="nil"/>' for s in
            ("top", "left", "bottom", "right", "insideH", "insideV")) + '</w:tblBorders>')
    no_indent = '<w:pPr><w:ind w:firstLine="0"/><w:jc w:val="{}"/></w:pPr>'
    out, pos, n = [], 0, 0
    for m, tag in zip(paras, tags):
        out.append(doc[pos:m.start()])
        pos = m.end()
        if tag is None:
            out.append(m.group(0))
            continue
        para = re.sub(r"^<w:p>(<w:pPr>.*?</w:pPr>)?", "<w:p>" + no_indent.format("center"),
                      m.group(0), count=1, flags=re.S)
        cell = ('<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:vAlign w:val="center"/>'
                '</w:tcPr>{body}</w:tc>')
        num = (f'<w:p>{no_indent.format("right")}<w:r><w:t>({tag})</w:t></w:r></w:p>')
        out.append(
            f'<w:tbl><w:tblPr><w:tblW w:w="{TEXT_W}" w:type="dxa"/>{none}'
            '<w:tblLayout w:type="fixed"/></w:tblPr>'
            f'<w:tblGrid><w:gridCol w:w="{TEXT_W - NUM_W}"/><w:gridCol w:w="{NUM_W}"/></w:tblGrid>'
            '<w:tr>' + cell.format(w=TEXT_W - NUM_W, body=para)
            + cell.format(w=NUM_W, body=num) + '</w:tr></w:tbl>')
        n += 1
    out.append(doc[pos:])
    parts["word/document.xml"] = "".join(out).encode("utf-8")
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return n


def to_doc(docx: Path) -> Path:
    """docx → doc через LibreOffice: формулы уходят объектами Equation 3.0."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise SystemExit("нет LibreOffice: apt-get install -y libreoffice-writer libreoffice-math")
    r = subprocess.run([soffice, "--headless", "--convert-to", "doc", "--outdir",
                        str(docx.parent), str(docx)], capture_output=True, text=True)
    dst = docx.with_suffix(".doc")
    if r.returncode != 0 or not dst.exists():
        raise SystemExit(f"[doc] LibreOffice не справился:\n{r.stderr[-1500:]}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--doc", action="store_true",
                    help="ещё и .doc: для Яндекс Документов, которые теряют формулы docx")
    a = ap.parse_args()

    if not shutil.which("pandoc"):
        raise SystemExit("нет pandoc: apt-get install -y pandoc")

    md, cut = cut_service_sections(Path(a.src).read_text())
    for marker in cut:
        print(f"   отрезан служебный раздел: {marker}")
    md = relocate(md)
    md, tags = strip_tags(md)

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

    n = number_equations(Path(a.out), tags)
    print(f"   номера формул проставлены: {n}")
    if a.doc:
        print(f"   .doc: {to_doc(Path(a.out))}")
    size = Path(a.out).stat().st_size // 1024
    print(f"готово: {a.out} ({size} КБ)")
    print("проверьте в Word: TNR 12, полуторный интервал, поля 25/25/25/15 мм,")
    print("формулы редактируемые (OMML), таблицы и рисунки после текста")


if __name__ == "__main__":
    main()
