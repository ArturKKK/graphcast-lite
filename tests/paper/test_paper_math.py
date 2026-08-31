"""Отрисовка формул рукописи.

Ошибки здесь тихие: формула просто печатается неправильно, а увидеть это можно
только глазами в готовом PDF. В рукописи, ушедшей на рецензию, стояло «50circ»
вместо «50°» — знак градуса терялся, и заметил это рецензент, а не мы.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_math import tex_to_html  # noqa: E402


def test_degree_sign_in_superscript():
    """`50^\\circ` — это 50°, а не «50circ».

    Разбор аргумента брал ровно одну литеру, и ею оказывалась обратная косая
    черта; имя команды уезжало в текст курсивом.
    """
    got = tex_to_html(r"\cos 50^\circ / \cos 60^\circ")
    assert "50<sup>°</sup>" in got
    assert "circ" not in got


def test_command_with_argument_stays_inside_the_superscript():
    """`A^\\mathsf{T}` — вся команда с аргументом внутри индекса.

    Иначе аргумент убегает наружу и печатается на строке: `A^T` превращается
    в «A» с пустым индексом и отдельно стоящим «T».
    """
    got = tex_to_html(r"A^\mathsf{T}")
    assert got.count("<sup>") == 1
    assert got.index("T") < got.index("</sup>"), "аргумент убежал из индекса"


def test_ordinary_superscripts_and_subscripts():
    assert tex_to_html(r"x^{2}") == "<i>x</i><sup>2</sup>"
    assert "<sup>−4</sup>" in tex_to_html(r"10^{-4}")
    got = tex_to_html(r"h_j^{(r)}")
    assert "<sub>" in got and "<sup>" in got


def test_greek_and_operators():
    assert "σ" in tex_to_html(r"\sigma_o^2")
    assert "≈" in tex_to_html(r"a \approx b")


def test_manuscript_formulas_render_without_unknown_commands():
    """Ни одна формула рукописи не даёт нераспознанной команды.

    Отрисовщик помечает такие места классом `unk`; в готовом PDF они выглядят
    как обрывок исходного кода посреди формулы.
    """
    import re
    art = ROOT / "docs" / "paper" / "article_gip.md"
    if not art.exists():
        return
    text = art.read_text()
    bad = []
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.S):
        html = tex_to_html(" ".join(m.group(1).split()))
        if 'class="unk"' in html:
            bad += re.findall(r'class="unk">([^<]*)<', html)
    for m in re.finditer(r"(?<!\$)\$([^$\n]+)\$(?!\$)", text):
        html = tex_to_html(m.group(1))
        if 'class="unk"' in html:
            bad += re.findall(r'class="unk">([^<]*)<', html)
    assert not bad, f"нераспознанные команды в формулах: {sorted(set(bad))}"
