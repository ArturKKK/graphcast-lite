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


def test_pdf_builder_drops_the_prompt(tmp_path):
    """Сборка, из которой печатается PDF, не тащит служебный раздел.

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
    assert "придирчивый рецензент" not in html, "промпт попал в вёрстку для PDF"
    assert "Wijnands" in html, "отрезано лишнее — пропал список литературы"
    assert "^1,2^" not in html, "крышки аффиляций не преобразованы"
    assert "<sup>1,2</sup>" in html


def test_service_section_list_is_not_empty():
    """Список служебных разделов не должен опустеть по недосмотру."""
    assert SERVICE_SECTIONS
    assert all(m.startswith("## ") for m in SERVICE_SECTIONS)
