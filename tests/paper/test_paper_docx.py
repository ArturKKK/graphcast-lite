"""Сборка рукописи в Word: требования журнала проверяются по самому файлу.

В редакцию уходит docx, а не PDF: «Текст набирается в формате Word шрифтом
Times New Roman 12 кеглем на листе форматом А4 с полями: нижнее, верхнее и
левое – 25 мм, правое – 15 мм» (Правила для авторов, п. 6). Проверить это
глазами нельзя — оформление живёт в XML внутри архива, — поэтому проверяем
разбором файла.

Отдельно про формулы: п. 15 требует редактор формул, а не картинки. pandoc
переводит $…$ в OMML, и тест следит, что формулы в документе действительно
разметкой, а не изображениями.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(shutil.which("pandoc") is None,
                                reason="нужен pandoc: apt-get install -y pandoc")

MM = 56.7
TWIP_25 = round(25 * MM)   # 1418
TWIP_15 = round(15 * MM)   # 850


@pytest.fixture(scope="module")
def docx(tmp_path_factory):
    out = tmp_path_factory.mktemp("docx") / "article.docx"
    r = subprocess.run([sys.executable, "scripts/paper_docx.py", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    z = zipfile.ZipFile(out)
    return {
        "zip": z,
        "doc": z.read("word/document.xml").decode(),
        "styles": z.read("word/styles.xml").decode(),
        "media": [n for n in z.namelist() if n.startswith("word/media/")],
        "path": out,
    }


def attrs(xml, tag):
    """Значения атрибутов тега — порядок в них pandoc не сохраняет."""
    m = re.search(rf"<{tag}\b([^>]*)/?>", xml)
    assert m, f"нет тега {tag}"
    return dict(re.findall(r'(\S+?)="([^"]*)"', m.group(1)))


def test_font_and_spacing(docx):
    st = docx["styles"]
    assert 'w:ascii="Times New Roman"' in st, "гарнитура не Times New Roman"
    assert 'w:sz w:val="24"' in st, "кегль не 12 (24 полупункта)"
    assert 'w:line="360"' in st, "интервал не полуторный (360 = 1,5 × 240)"
    assert 'w:jc w:val="both"' in st, "выравнивание не по ширине"
    assert 'w:firstLine="567"' in st, "абзацный отступ не 1 см"


def test_page_size_and_margins(docx):
    sz = attrs(docx["doc"], "w:pgSz")
    assert int(sz["w:w"]) == pytest.approx(11906, abs=4), "лист не A4 по ширине"
    assert int(sz["w:h"]) == pytest.approx(16838, abs=4), "лист не A4 по высоте"
    mar = attrs(docx["doc"], "w:pgMar")
    for side in ("w:top", "w:bottom", "w:left"):
        assert int(mar[side]) == pytest.approx(TWIP_25, abs=2), f"{side} не 25 мм"
    assert int(mar["w:right"]) == pytest.approx(TWIP_15, abs=2), "правое поле не 15 мм"


def test_formulas_are_editable_not_pictures(docx):
    """П. 15: формулы через редактор формул. OMML — родной формат Word."""
    assert docx["doc"].count("<m:oMath") > 20, "формул OMML нет — они стали картинками?"


def test_all_tables_present(docx):
    from paper_artifact import TABLES_EN
    assert docx["doc"].count("<w:tbl>") == len(TABLES_EN)


def test_figures_embedded(docx):
    from paper_artifact import FIGURES
    assert len(docx["media"]) == len(FIGURES), f"картинок {len(docx['media'])}"


def test_tables_and_figures_come_after_the_text(docx):
    """П. 13, 14: и те, и другие — после основного текста."""
    text = re.sub(r"<[^>]+>", " ", docx["doc"])
    refs = text.index("Список литературы")
    first_tbl = docx["doc"].index("<w:tbl>")
    text_before_tbl = len(re.sub(r"<[^>]+>", " ", docx["doc"][:first_tbl]).split())
    assert refs > 0, "в документе нет списка литературы"
    # Первая таблица должна стоять дальше основного текста: слов до неё
    # заметно больше, чем в одном разделе.
    assert text_before_tbl > 5000, (
        f"первая таблица слишком рано: до неё всего {text_before_tbl} слов")


def test_service_sections_are_cut(docx):
    """Промпт для рецензирующей нейросети в редакцию уехать не должен."""
    text = re.sub(r"<[^>]+>", " ", docx["doc"])
    for bad in ("ЗАПОЛНИТЬ", "промпт", "ЧЕРНОВИК"):
        assert bad not in text, f"в рукописи осталось служебное: {bad}"


def test_superscript_affiliations(docx):
    """Цифры аффилиации — надстрочные, а не «^1,2^» буквально."""
    text = re.sub(r"<[^>]+>", " ", docx["doc"])
    assert "^1" not in text and "^2^" not in text, "крышки остались в тексте"
    assert "superscript" in docx["doc"], "надстрочных знаков нет вовсе"
