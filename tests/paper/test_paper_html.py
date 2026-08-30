"""Сборка рукописи в HTML: надстрочные знаки и сохранность формул.

Ошибка тут тихая и стыдная: разметка уезжает в вёрстку буквально, и это видно
только глазами в готовом файле. До 30.08.2026 в рукописи, идущей в журнал,
стояло «Табаков^1,2^» вместо надстрочных единицы и двойки.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "paper_to_html.py"
ARTICLE = ROOT / "docs" / "paper" / "article_gip.md"


def build(tmp_path, text):
    src = tmp_path / "a.md"
    src.write_text(text)
    out = tmp_path / "a.html"
    r = subprocess.run([sys.executable, str(SCRIPT), "--src", str(src), "--out", str(out)],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return out.read_text(), r.stdout


def test_affiliation_superscripts_become_tags(tmp_path):
    html, _ = build(tmp_path, "А.С. Табаков^1,2^, А.В. Пененко^1^\n\n^2^НГУ\n")
    assert "Табаков<sup>1,2</sup>" in html
    assert "Пененко<sup>1</sup>" in html
    assert "<sup>2</sup>НГУ" in html
    assert "^1," not in html and "^2^" not in html


def test_latex_superscripts_are_left_alone(tmp_path):
    """Формулы не должны пострадать от замены.

    Это главная опасность правки: широкий шаблон испортил бы всю математику
    статьи молча — формулы просто стали бы неверными.
    """
    math = r"$h_j^{(r)} = \phi\left(x^{2}\right)^{\mathsf{T}}$ и $10^{-4}$"
    html, _ = build(tmp_path, math + "\n")
    for frag in ("^{(r)}", "^{2}", "^{\\mathsf{T}}", "^{-4}"):
        assert frag in html, f"формула повреждена: пропало {frag}"
    assert "<sup>" not in html


def test_real_manuscript_has_no_leftover_carets():
    """В самой рукописи после сборки не остаётся сырых крышек аффиляций."""
    if not ARTICLE.exists():
        return
    import re
    text = ARTICLE.read_text()
    # то же преобразование, что и в сборщике
    converted = re.sub(r"\^(\d+(?:,\d+)*)\^", r"<sup>\1</sup>", text)
    assert re.search(r"\^\d+(?:,\d+)*\^", converted) is None
    assert converted.count("<sup>") >= 6, "аффиляции не найдены — изменилась шапка?"


def test_conversion_count_is_reported(tmp_path):
    """Число замен печатается — иначе молчаливый ноль не отличить от работы."""
    _, out = build(tmp_path, "Автор^1^\n\n^1^Институт\n")
    assert "надстрочных знаков аффиляций: 2" in out


def test_review_prompt_never_reaches_the_manuscript(tmp_path):
    """Служебный раздел с промптом для рецензирующей нейросети отрезается всегда.

    Он лежит в том же файле, чтобы не потеряться, но подать его в журнал вместе
    со статьёй нельзя. Отсечение НЕ должно зависеть от флага: 30.08.2026 я
    полагался на разрез по «## Перечень рисунков», которого в рукописи вообще
    нет, и промпт спокойно попал бы в PDF.
    """
    text = ("# Статья\n\nТекст рукописи.\n\n"
            "## Промпт для независимой проверки рукописи\n\n"
            "Ты — придирчивый рецензент журнала.\n")
    html, out = build(tmp_path, text)
    assert "Текст рукописи" in html
    assert "придирчивый рецензент" not in html
    assert "отрезан служебный раздел" in out


def test_real_manuscript_build_drops_the_prompt(tmp_path):
    """То же на настоящей рукописи, а не на выдуманном куске."""
    if not ARTICLE.exists():
        return
    out = tmp_path / "real.html"
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--src", str(ARTICLE), "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    html = out.read_text()
    assert "придирчивый рецензент" not in html, "промпт попал в рукопись"
    assert "Список литературы" in html, "отрезано лишнее — пропал список литературы"
