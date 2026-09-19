"""Механически проверяемые требования «Правил для авторов» ГИиП.

Правила скачаны с meteoinfo.ru (rules-for-authors.pdf, редакция 2025 г.).
Проверяется то, что можно проверить кодом; вёрстку — поля, кегль, интервал —
сторожит scripts/paper_pdf.sh, а объём в 20 полос виден в его выводе.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

ARTICLE = ROOT / "docs" / "paper" / "article_gip.md"


@pytest.fixture(scope="module")
def body():
    """Текст без списков литературы: в них своя пунктуация по ГОСТ."""
    s = ARTICLE.read_text()
    return s[:s.index("## Список литературы")]


def abstracts():
    lines = ARTICLE.read_text().splitlines()
    ru = next(x for x in lines if x.startswith("Представлена "))
    en = next(x for x in lines if x.startswith("A regional "))
    return ru, en


@pytest.mark.parametrize("lang,text", list(zip(("рус.", "англ."), abstracts())))
def test_abstract_length(lang, text):
    """П. 9: аннотация 100–200 слов."""
    n = len(text.split())
    assert 100 <= n <= 200, f"{lang} аннотация {n} слов"


@pytest.mark.parametrize("head", ["Ключевые слова:", "Keywords:"])
def test_keyword_count(head):
    """П. 11: 5–10 понятий."""
    line = next(x for x in ARTICLE.read_text().splitlines() if x.startswith(head))
    n = len(line[len(head):].strip().rstrip(".").split(","))
    assert 5 <= n <= 10, f"{head} {n} шт."


def test_udk_present():
    assert ARTICLE.read_text().lstrip().startswith("УДК "), "нет индекса УДК"


def test_no_hyphen_instead_of_dash_in_ranges(body):
    """П. 12: «Не допускается использовать дефис (-) вместо знака тире (–)»."""
    bad = re.findall(r"\d+-\d+(?!\d)", body)
    assert not bad, f"дефис в диапазонах: {sorted(set(bad))[:5]}"


def test_space_between_value_and_unit(body):
    """П. 12: «Между цифрой и единицей измерения вставляется один пробел»."""
    bad = re.findall(r"\d(?:%|°C|м/с|гПа|гпм|км(?![а-я]))", body)
    assert not bad, f"слиплись: {sorted(set(bad))[:5]}"


def test_guillemets_only(body):
    """П. 12: кавычки «…»."""
    assert not re.findall(r'"[^"\n]{1,60}"', body), "в тексте кавычки-лапки"


def test_no_leftover_citation_brackets(body):
    """Удаление источника не должно оставлять пустых скобок или лишних пробелов."""
    assert not re.findall(r"\[\s*\]|\[\s*,|,\s*\]", body), "пустая ссылка"
    assert not re.findall(r"\s+[,.;:](?:\s|$)", body), "пробел перед знаком препинания"


def test_table_captions_are_numbered_in_order():
    """П. 13: «Таблица 1. Название таблицы»."""
    nums = [int(n) for n in re.findall(r"\*\*Таблица (\d+)\.\*\*", ARTICLE.read_text())]
    assert nums == list(range(1, len(nums) + 1)), f"нумерация таблиц: {nums}"


def test_russian_sources_come_first():
    """Правила списка литературы: сначала русские издания, затем иностранные."""
    s = ARTICLE.read_text()
    block = s[s.index("## Список литературы"):s.index("## References")]
    cyrillic = [bool(re.match(r"\d+\.\s\*[А-ЯЁ]", line))
                for line in block.splitlines() if re.match(r"\d+\.\s", line)]
    assert cyrillic, "список литературы пуст"
    first_latin = cyrillic.index(False) if False in cyrillic else len(cyrillic)
    assert not any(cyrillic[first_latin:]), "русский источник после иностранных"


def test_surnames_are_italic():
    """П. 12: «Фамилии выделяются курсивом»."""
    s = ARTICLE.read_text()
    block = s[s.index("## Список литературы"):s.index("## References")]
    entries = [x for x in block.splitlines() if re.match(r"\d+\.\s", x)]
    assert entries
    plain = [e[:40] for e in entries if not re.match(r"\d+\.\s\*", e)]
    assert not plain, f"фамилия без курсива: {plain}"
