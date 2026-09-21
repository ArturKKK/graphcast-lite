"""Рисунки и таблицы должны попадать в рукопись, а не пропадать молча.

Вставка рисунка идёт по опорной фразе из текста статьи. 19.09.2026 очередная
правка текста задела эту фразу, сборка напечатала предупреждение и собрала PDF
без рис. 2 — с кодом возврата 0 и подписью «готово». В журнал ушла бы рукопись
со ссылкой «см. рис. 2» и без самого рисунка.

Здесь же проверяется перенос таблиц и рисунков в конец: правила журнала (п. 13,
14) требуют размещать их на отдельных страницах после основного текста и давать
подписи на русском и английском языках.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import paper_artifact  # noqa: E402


@pytest.fixture(scope="module")
def built():
    """Сборка НАСТОЯЩЕЙ рукописи: фразы-опоры и таблицы живут именно в ней."""
    return paper_artifact.build()


def test_anchors_present_in_article():
    md = paper_artifact.SRC.read_text()
    missing = [a for _, a, _, _ in paper_artifact.FIGURES if a not in md]
    assert not missing, f"опорные фразы потеряны: {missing}"


def test_every_figure_lands_in_html(built):
    for name, _, caption, caption_en in paper_artifact.FIGURES:
        assert caption in built, f"{name}: русской подписи нет в собранном HTML"
        assert caption_en in built, f"{name}: английской подписи нет"
        assert built.count(caption) == 1, f"{name}: подпись вставлена дважды"


def test_figure_svg_is_inlined(built):
    """Именно inline-SVG: внешних файлов вёрстка не тянет."""
    assert built.count('<figure class="fig">') == len(paper_artifact.FIGURES)
    assert "<svg" in built


def test_every_table_has_english_caption(built):
    """Пропущенная английская подпись — прямое нарушение п. 5 правил."""
    numbers = re.findall(r"\*\*(Таблица \d+\.)\*\*", paper_artifact.SRC.read_text())
    assert numbers, "в статье не нашлось ни одной подписи таблицы"
    missing = [n for n in numbers if n not in paper_artifact.TABLES_EN]
    assert not missing, f"нет английской подписи для {missing}"
    for n in numbers:
        assert paper_artifact.TABLES_EN[n] in built, f"{n}: англ. подпись не вставлена"


def test_tables_and_figures_move_after_the_text(built):
    """Ни одной таблицы и ни одного рисунка не должно остаться в тексте."""
    tail = built.index('<div class="tail">')
    head = built[:tail]
    assert "<table>" not in head, "таблица осталась в основном тексте"
    assert '<figure class="fig">' not in head, "рисунок остался в основном тексте"
    # а ссылки на них из текста — остались
    assert "табл. 1" in head and "рис. 2" in head


def test_broken_anchor_is_fatal(monkeypatch):
    """Сорванная опора обязана ронять сборку, а не печатать предупреждение."""
    name, _, caption, caption_en = paper_artifact.FIGURES[-1]
    monkeypatch.setattr(paper_artifact, "FIGURES",
                        [(name, "такой фразы в статье заведомо нет", caption, caption_en)])
    with pytest.raises(SystemExit) as e:
        paper_artifact.build()
    assert name in str(e.value)


def test_missing_table_caption_is_fatal(monkeypatch):
    """Подпись таблицы, которой нет в статье, тоже обязана ронять сборку."""
    # Именно ДОБАВЛЯЕМ: подмена всего словаря сработала бы на другой проверке —
    # на той, что ловит таблицу без английской подписи.
    monkeypatch.setattr(paper_artifact, "TABLES_EN",
                        {**paper_artifact.TABLES_EN, "Таблица 99.": "Table 99."})
    with pytest.raises(SystemExit) as e:
        paper_artifact.build()
    assert "Таблица 99." in str(e.value)
