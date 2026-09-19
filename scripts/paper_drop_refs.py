#!/usr/bin/env python3
"""Удаление источников из обоих списков литературы с пересчётом нумерации.

Списка два — русский и латинский References, — и они обязаны совпадать по
номерам и порядку. Вручную это делалось трижды и трижды расходилось: то номер
в тексте указывал на соседа, то запись оставалась в одном списке и пропадала в
другом. Скрипт правит оба списка и все ссылки в тексте одним проходом.

Порядок в обоих списках алфавитный по фамилии (правила журнала, п. 12), поэтому
после удаления номера просто сдвигаются — переупорядочивать не нужно.

  python scripts/paper_drop_refs.py 1 5 14          # показать, что получится
  python scripts/paper_drop_refs.py 1 5 14 --apply  # записать
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "paper" / "article_gip.md"
LISTS = ("## Список литературы", "## References")


def split_list(text, head):
    """Возвращает (начало, конец) блока записей списка."""
    i = text.index(head) + len(head)
    j = len(text)
    for other in LISTS + ("\n---",):
        k = text.find(other, i)
        if k != -1:
            j = min(j, k)
    return i, j


def entries(block):
    """Записи списка: [(номер, текст записи с переводами строк)]."""
    out = []
    for m in re.finditer(r"^(\d+)\.\s", block, flags=re.M):
        start = m.start()
        nxt = re.search(r"^\d+\.\s", block[m.end():], flags=re.M)
        end = m.end() + nxt.start() if nxt else len(block)
        out.append((int(m.group(1)), block[start:end]))
    return out


def renumber_citations(text, mapping):
    """[7, 11] -> новые номера; ссылку на удалённый источник убираем из группы."""
    def one(m):
        kept = [str(mapping[int(n)]) for n in re.split(r"\s*,\s*", m.group(1))
                if int(n) in mapping]
        return f"[{', '.join(kept)}]" if kept else "\x00"
    text = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", one, text)
    # Вырезанная целиком ссылка оставляет за собой лишний пробел: «модели ;».
    return re.sub(r" *\x00", "", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drop", type=int, nargs="+", help="номера удаляемых источников")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    drop = set(a.drop)

    text = SRC.read_text()
    body_end = text.index(LISTS[0])
    body = text[:body_end]

    blocks = {}
    for head in LISTS:
        i, j = split_list(text, head)
        blocks[head] = (i, j, entries(text[i:j]))

    nums = [n for n, _ in blocks[LISTS[0]][2]]
    for head, (_, _, es) in blocks.items():
        if [n for n, _ in es] != nums:
            sys.exit(f"списки разошлись по номерам: {head}")
    missing = drop - set(nums)
    if missing:
        sys.exit(f"нет таких источников: {sorted(missing)}")

    mapping, k = {}, 0
    for n in nums:
        if n in drop:
            continue
        k += 1
        mapping[n] = k

    still = set()
    for m in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", body):
        still |= {int(x) for x in re.split(r"\s*,\s*", m.group(1))}
    orphan = drop & still
    print(f"удаляем {sorted(drop)}, остаётся {len(mapping)} источников")
    if orphan:
        print(f"ВНИМАНИЕ: на {sorted(orphan)} ещё есть ссылки в тексте — "
              f"они будут просто вырезаны, проверьте фразы глазами")

    out = renumber_citations(body, mapping)
    for head in LISTS:
        i, j, es = blocks[head]
        kept = "".join(re.sub(r"^\d+\.", f"{mapping[n]}.", e, count=1)
                       for n, e in es if n not in drop)
        out += head + "\n\n" + kept.strip() + "\n\n"

    if a.apply:
        SRC.write_text(out)
        print(f"записано: {SRC}")
    else:
        print("пробный прогон; для записи добавьте --apply")
        for old, new in sorted(mapping.items()):
            if old != new:
                print(f"   [{old}] -> [{new}]")


if __name__ == "__main__":
    main()
