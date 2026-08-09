#!/usr/bin/env python3
"""Собирает единую рукопись статьи из шапки и разделов по формату журнала
«Гидрометеорологические исследования и прогнозы».

Что делает:
  * склеивает docs/paper/00_frontmatter.md и docs/paper/sections/*.md по порядку;
  * УБИРАЕТ нумерацию разделов («## 3.2. Архитектура» → «Архитектура»),
    поскольку в журнале разделы не нумеруются (сверено с вып. № 1 (395), 2025);
  * вырезает служебные блоки разделов (Рисунки / Таблицы / Литература / Замечания автору)
    и собирает их в конец документа отдельными перечнями;
  * строит единый «Список литературы» без дублей и заготовку блока «References»;
  * считает объём и все оставшиеся плейсхолдеры {{ЗАПОЛНИТЬ}}.

Использование:
    python scripts/paper_assemble.py                      # → docs/paper/article_gip.md
    python scripts/paper_assemble.py --out other.md
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "paper"
SECTIONS_ORDER = [
    "01_intro.md",
    "02_data.md",
    "03_model.md",
    "04_results.md",
    "05_assim.md",
    "07_discussion.md",
]
SERVICE_HEADINGS = ("Рисунки", "Таблицы", "Литература", "Замечания автору")


def strip_numbering(line: str) -> str:
    """«## 3.2. Архитектура» → «## Архитектура»; «# 4. Результаты: …» → «# Результаты: …»."""
    return re.sub(r"^(#{1,6})\s+\d+(?:\.\d+)*\.?\s*", r"\1 ", line)


def split_service(text: str):
    """Отделяет основной текст от служебных блоков (рисунки/таблицы/литература/замечания)."""
    body, service = [], {}
    current = None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m and any(m.group(1).startswith(h) for h in SERVICE_HEADINGS):
            current = m.group(1)
            service.setdefault(current, [])
            continue
        if current and re.match(r"^#\s+", line):   # новый раздел верхнего уровня
            current = None
        (service[current] if current else body).append(line)
    return "\n".join(body), {k: "\n".join(v).strip() for k, v in service.items()}


def parse_refs(block: str) -> list[str]:
    """Из блока литературы достаёт записи по нумерации «1. …»."""
    out, buf = [], None
    for line in block.splitlines():
        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            if buf:
                out.append(" ".join(buf).strip())
            buf = [m.group(1)]
        elif buf is not None and line.strip():
            buf.append(line.strip())
    if buf:
        out.append(" ".join(buf).strip())
    return out


def ref_key(ref: str) -> str:
    """Ключ для дедупликации: первые авторы + начало названия, без регистра и пунктуации."""
    s = re.sub(r"\{\{.*?\}\}", "", ref)
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return " ".join(s.split()[:8])


def is_cyrillic(ref: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", ref[:60]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PAPER / "article_gip.md"))
    a = ap.parse_args()

    parts, figures, tables, notes = [], [], [], []
    refs_all: list[str] = []
    seen: set[str] = set()

    fm = (PAPER / "00_frontmatter.md").read_text()
    parts.append(fm.split("<!--")[0].rstrip())

    for fn in SECTIONS_ORDER:
        p = PAPER / "sections" / fn
        if not p.exists():
            print(f"[warn] нет {fn} — пропущен")
            continue
        raw = p.read_text()
        raw = re.sub(r"^_\(оценка:.*?\)_\s*$", "", raw, flags=re.M)   # служебная строка объёма
        body, service = split_service(raw)
        body = "\n".join(strip_numbering(l) for l in body.splitlines())
        # первый заголовок раздела делаем h1, вложенные — h2
        body = re.sub(r"^#\s+", "\n## ", body, flags=re.M)
        body = re.sub(r"^###\s+", "### ", body, flags=re.M)
        parts.append(body.strip())

        sec_name = fn[3:-3]
        for key, dst in (("Рисунки", figures), ("Таблицы", tables), ("Замечания автору", notes)):
            for k, v in service.items():
                if k.startswith(key) and v:
                    dst.append(f"**{sec_name}:**\n{v}")
        for k, v in service.items():
            if k.startswith("Литература"):
                for r in parse_refs(v):
                    kk = ref_key(r)
                    if kk and kk not in seen:
                        seen.add(kk)
                        refs_all.append(r)

    # ---- список литературы: русские источники, затем латиница (как в журнале) ----
    cyr = [r for r in refs_all if is_cyrillic(r)]
    lat = [r for r in refs_all if not is_cyrillic(r)]
    refs_sorted = cyr + lat

    lit = ["## Список литературы", ""]
    lit += [f"{i}. {r}" for i, r in enumerate(refs_sorted, 1)]
    lit += [
        "", "## References", "",
        "{{ЗАПОЛНИТЬ: транслитерированный список по правилам журнала — русские работы: "
        "Author I.O. Perevod nazvaniya. Transliteraciya zhurnala [Journal in English], "
        "год, vol., pp. [in Russ.]; латинские записи переносятся с заменой «//» на «.» "
        "и «С.» на «pp.». Формат сверен с вып. № 1 (395), 2025}}",
    ]

    doc = "\n\n".join(parts) + "\n\n" + "\n".join(lit)

    doc += "\n\n---\n\n## Перечень рисунков (в отдельные файлы .jpg/.tif)\n\n" + "\n\n".join(figures)
    doc += "\n\n## Перечень таблиц\n\n" + "\n\n".join(tables)
    doc += "\n\n## Замечания и решения для авторов\n\n" + "\n\n".join(notes)

    out = Path(a.out)
    out.write_text(doc)

    # ---- сводка ----
    words = len(re.findall(r"\b[\w-]+\b", doc))
    ph = re.findall(r"\{\{ЗАПОЛНИТЬ[^}]*\}\}", doc)
    print(f"→ {out}")
    print(f"   слов: {words}  ·  ≈{words/350:.1f} стр. (TNR 12, инт. 1.5, ~350 слов/стр.)")
    print(f"   ссылок в списке: {len(refs_sorted)} (рус. {len(cyr)} + лат. {len(lat)})")
    print(f"   рисунков: {len(figures)} блоков, таблиц: {len(tables)} блоков")
    print(f"   незаполненных плейсхолдеров: {len(ph)}")
    for x in ph[:12]:
        print("     ·", x[:110])


if __name__ == "__main__":
    main()
