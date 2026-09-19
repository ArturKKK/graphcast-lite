"""Перекрёстные ссылки внутри рукописи должны указывать на существующие пункты.

Разделы не нумерованы в исходнике — номер получается из порядка заголовков.
Поэтому удаление одного подпункта молча сдвигает нумерацию всех следующих, и
ссылка «см. п. 4.7» начинает указывать в пустоту. Так и вышло 19.09.2026, когда
п. 4.6 влили в п. 4.5.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs" / "paper" / "article_gip.md"

# Заголовки верхнего уровня: открывают раздел, сами номера подпункта не получают.
PARTS = (
    "Данные, базовая модель и метрики качества",
    "Мультимасштабная графовая модель и стратегия дообучения",
    "Результаты: качество регионального прогноза",
    "Обсуждение, выводы и направления дальнейшей работы",
)
APPENDIX = ("Список литературы", "References")


def numbering(text):
    sec, sub, out = 1, 0, {}
    for line in text.splitlines():
        if not line.startswith("## "):
            continue
        title = line[3:].strip()
        if title in PARTS:
            sec, sub = sec + 1, 0
        elif title not in APPENDIX:
            sub += 1
            out[f"{sec}.{sub}"] = title
    return out


def test_every_cross_reference_resolves():
    text = ARTICLE.read_text()
    known = numbering(text)
    used = set(re.findall(r"п\. (\d+\.\d+)", text))
    assert used, "в статье вообще не осталось перекрёстных ссылок — проверка ослепла"
    missing = sorted(used - set(known))
    assert not missing, f"ссылки в никуда: {missing}; есть пункты {sorted(known)}"


def test_no_self_reference():
    """Пункт не должен ссылаться сам на себя — верный признак съехавшей нумерации."""
    text = ARTICLE.read_text()
    known = numbering(text)
    by_title = {t: n for n, t in known.items()}
    bad = []
    for chunk in re.split(r"^## ", text, flags=re.M)[1:]:
        title = chunk.splitlines()[0].strip()
        here = by_title.get(title)
        if here and f"п. {here}" in chunk:
            bad.append(title)
    assert not bad, f"пункт ссылается на самого себя: {bad}"
