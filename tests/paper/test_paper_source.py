"""Предобработка исходника рукописи — общая для обеих сборок.

Модуль появился потому, что сборок две: `paper_to_html.py` для вычитки и
`paper_artifact.py`, из которой печатается PDF. 30.08.2026 правки внесли только
в первую, и та сборка, что идёт в журнал, осталась и с крышками вместо
надстрочных знаков, и с приложенным промптом для рецензирующей нейросети.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_source import SERVICE_SECTIONS, prepare, superscripts  # noqa: E402


def test_affiliation_superscripts():
    got, n = superscripts("Табаков^1,2^, Пененко^1^")
    assert got == "Табаков<sup>1,2</sup>, Пененко<sup>1</sup>"
    assert n == 2


def test_math_is_untouched():
    """Главная опасность замены: широкий шаблон испортил бы всю математику."""
    math = r"$h_j^{(r)}$ и $x^{2}$ и $10^{-4}$ и $A^{\mathsf{T}}$"
    got, n = superscripts(math)
    assert got == math
    assert n == 0


def test_service_sections_are_cut():
    text = "Статья.\n\n## Промпт для независимой проверки рукописи\n\nТы рецензент.\n"
    got, cut = prepare(text)
    assert "Ты рецензент" not in got
    assert "Статья." in got
    assert cut["cut"] == ["## Промпт для независимой проверки"]


def test_nothing_is_cut_when_there_is_nothing_to_cut():
    got, done = prepare("Просто текст.\n")
    assert done["cut"] == []
    assert got.strip() == "Просто текст."


def test_both_builders_use_the_shared_module():
    """Обе сборки зовут общий модуль, а не свои копии правил.

    Проверка против повторения истории: копия правила в одном скрипте
    расходится с оригиналом в другом, и расхождение всплывает в вёрстке.
    """
    for name in ("paper_to_html.py", "paper_artifact.py"):
        src = (ROOT / "scripts" / name).read_text()
        assert "from paper_source import prepare" in src, name
        assert "<sup>" not in src, f"{name}: своя копия правила надстрочных знаков"


def test_pdf_builder_output_is_sane(tmp_path):
    """Сборка, из которой печатается PDF, даёт годную вёрстку.

    Это тот самый путь, который уходит в журнал, и проверять надо именно его,
    а не соседнюю сборку для вычитки.
    """
    article = ROOT / "docs" / "paper" / "article_gip.md"
    artifact = ROOT / "docs" / "paper" / "artifact.html"
    if not article.exists():
        return
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "paper_artifact.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    html = artifact.read_text()
    assert "Wijnands" in html, "пропал список литературы"
    assert "^1,2^" not in html, "крышки аффиляций не преобразованы"
    assert "<sup>1,2</sup>" in html


def test_review_prompt_lives_outside_the_manuscript():
    """Промпт для рецензирующей нейросети — в отдельном файле, не в статье.

    Раньше он лежал в конце рукописи и держался только на отсечении при сборке.
    Отдельный файл надёжнее: попасть в статью он физически не может.
    """
    article = ROOT / "docs" / "paper" / "article_gip.md"
    prompt = ROOT / "docs" / "paper" / "review_prompt.md"
    assert prompt.exists(), "файл с промптом пропал"
    body = prompt.read_text()
    assert "придирчивый рецензент" in body
    assert len(body) > 3000, "промпт подозрительно короткий — не обрезан ли?"
    if article.exists():
        assert "придирчивый рецензент" not in article.read_text(), (
            "промпт вернулся в рукопись")


def test_cutting_still_guards_against_the_prompt_coming_back():
    """Отсечение оставлено страховкой на случай, если промпт снова впишут.

    Файл вынесен, но правило не убрано: стоит дешёво, а цена отказа — служебный
    текст в журнальной подаче.
    """
    text = "Статья.\n\n## Промпт для независимой проверки\n\nТы рецензент.\n"
    got, done = prepare(text)
    assert "Ты рецензент" not in got
    assert done["cut"]


def test_service_section_list_is_not_empty():
    """Список служебных разделов не должен опустеть по недосмотру."""
    assert SERVICE_SECTIONS
    assert all(m.startswith("## ") for m in SERVICE_SECTIONS)


def test_list_indent_leaves_room_for_two_digit_markers():
    """Отступ списка вмещает маркер «11.», а не обрезает его.

    При padding-left 1,2em маркер выносился влево за печатную область и
    обрезался по краю страницы: у списка выводов от «1.» оставалась одна точка,
    а двузначные номера списка литературы пропали бы целиком. Проверять это
    глазами каждый раз нельзя, поэтому значение закреплено здесь.
    """
    import re
    css = (ROOT / "scripts" / "paper_artifact.py").read_text()
    m = re.search(r"ul,\s*ol\s*\{[^}]*padding-left:\s*([\d.]+)em", css)
    assert m, "не нашёл отступ списков в стилях"
    assert float(m.group(1)) >= 2.0, (
        f"отступ {m.group(1)}em мал для маркера «11.» — номера обрежутся")
