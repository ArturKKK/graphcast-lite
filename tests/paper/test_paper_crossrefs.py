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


def references():
    """(номера в русском списке, номера в References, номера, цитируемые в тексте)."""
    text = ARTICLE.read_text()
    ru_at, en_at = text.index("## Список литературы"), text.index("## References")
    body = text[:ru_at]
    cited = set()
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
        cited |= {int(x) for x in re.split(r"\s*,\s*", m.group(1))}
    return numbers(text[ru_at:en_at]), numbers(text[en_at:]), cited


def numbers(block):
    return {int(m.group(1)) for m in re.finditer(r"^(\d+)\.\s", block, re.M)}


def test_both_reference_lists_agree():
    """Списка два, и они обязаны совпадать по номерам (правила, п. 12)."""
    ru, en, _ = references()
    assert ru == en, (f"расходятся: только в рус. {sorted(ru - en)}, "
                      f"только в References {sorted(en - ru)}")


def test_no_orphan_or_dangling_references():
    """Ни записи без ссылки, ни ссылки без записи.

    19.09.2026 удаление абзаца «Обсуждения» разом унесло три ссылки [8, 9, 13];
    останься они единственными, записи повисли бы в списке без упоминания.
    """
    ru, _, cited = references()
    assert not (ru - cited), f"в списке есть, но нигде не цитируются: {sorted(ru - cited)}"
    assert not (cited - ru), f"цитируются, но записи нет: {sorted(cited - ru)}"


def test_reference_numbers_are_contiguous():
    ru, _, _ = references()
    assert ru == set(range(1, len(ru) + 1)), f"дыра в нумерации: {sorted(ru)}"


def test_saved_run_metrics_are_actually_tracked():
    """Посрочные метрики обязаны лежать в git, несмотря на *.npz в .gitignore.

    20.09.2026 новые прогоны не доехали в репозиторий: scripts/_vm_save_runs.sh
    делал `git add` без -f, gitignore отсекал их молча, и скрипт рапортовал
    «нового нет». Проверяем, что и раньше сохранённое отслеживается, и что в
    скрипте стоит -f.
    """
    import subprocess
    tracked = subprocess.run(
        ["git", "ls-files", "docs/paper/runs"], cwd=ROOT,
        capture_output=True, text=True).stdout.splitlines()
    npz = [f for f in tracked if f.endswith("_samples.npz")]
    assert len(npz) > 100, f"посрочных метрик в git всего {len(npz)} — их вымыло gitignore"

    saver = (ROOT / "scripts" / "_vm_save_runs.sh").read_text()
    assert "git add -f" in saver, "без -f новые прогоны в репозиторий не попадут"


def test_run_saver_covers_every_batch_prefix():
    """Сохранялка должна знать про все батчи, а не только про первый.

    21.09.2026 scripts/_vm_save_runs.sh копировал только файлы с префиксом w_,
    и результаты батча заморозки (f_) молча не доехали: скрипт отчитался
    «скопировано 12, нового нет».
    """
    import re
    saver = (ROOT / "scripts" / "_vm_save_runs.sh").read_text()
    m = re.search(r'PREFIXES=\$\{PREFIXES:-"([^"]+)"\}', saver)
    assert m, "в сохранялке нет списка префиксов"
    known = set(m.group(1).split())

    # Каждый батч кладёт npz с собственным префиксом — собираем их из раннеров.
    used = set()
    for sh in (ROOT / "scripts").glob("_vm_batch_*.sh"):
        for tag in re.findall(r"^\s*run\s+([a-z0-9]+)_", sh.read_text(), re.M):
            used.add(tag + "_")
    assert used, "не нашёл ни одного раннера с прогонами"
    assert used <= known, f"сохранялка не знает про префиксы {sorted(used - known)}"
