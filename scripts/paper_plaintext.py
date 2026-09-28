#!/usr/bin/env python3
"""Текст статьи для проверки стиля: только проза, без разметки.

Сервисы проверки на «слоп» и Главред принимают обычный текст. Таблицы,
выключные формулы, списки литературы и английская аннотация им только
мешают, поэтому выбрасываются. Формулы в строке переводятся в обычные символы
(τ, φ, x̂), подписи таблиц и рисунков остаются: это тоже текст статьи.

  python3 scripts/paper_plaintext.py            # → docs/paper/article_gip_text.md
"""
import argparse
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from paper_math import tex_to_html  # noqa: E402

SRC = ROOT / "docs" / "paper" / "article_gip.md"
OUT = ROOT / "docs" / "paper" / "article_gip_text.md"


def tex_plain(tex: str) -> str:
    h = tex_to_html(tex)
    h = h.replace('<span class="ovl">', "").replace("</i></span>", "̄")
    h = re.sub(r"<sub>(.*?)</sub>", r"\1", h)
    h = re.sub(r"<sup>(.*?)</sup>", r"^\1", h)
    h = re.sub(r"<[^>]+>", "", h)
    return html.unescape(h).replace(" ", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t = SRC.read_text()
    t = t[:t.index("## Список литературы")]
    # шапка (УДК, авторы, организации, почта) проверке стиля не нужна
    title = re.search(r"^# (.+)$", t, flags=re.M).group(1)
    t = f"# {title}\n\n" + t[t.index("Статья описывает"):]
    # английская часть: от английского заголовка до «Введения»
    t = re.sub(r"^# A multiscale.*?(?=^## Введение)", "", t, flags=re.S | re.M)
    t = re.sub(r"\$\$.*?\$\$", "", t, flags=re.S)
    t = re.sub(r"\$([^$\n]+)\$", lambda m: tex_plain(m.group(1)), t)
    t = "\n".join(ln for ln in t.splitlines() if not ln.startswith("|"))
    t = t.replace("[ТАБЛИЦА]", "")
    t = re.sub(r"\^([^^\s]+)\^", r"\1", t)          # надстрочные ^1,2^ у авторов
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"^---\s*$", "", t, flags=re.M)
    t = re.sub(r"\n{3,}", "\n\n", t).strip() + "\n"
    Path(a.out).write_text(t)
    print(f"готово: {a.out} — {len(t.split())} слов")


if __name__ == "__main__":
    main()
