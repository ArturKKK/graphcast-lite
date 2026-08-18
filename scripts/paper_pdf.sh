#!/usr/bin/env bash
# Собирает PDF рукописи: markdown -> страница -> печать в A4.
#
# Движок — wkhtmltopdf: chromium на этой машине оказался заглушкой от snap,
# а weasyprint требует pango и cairo, которых нет. Старый WebKit внутри
# wkhtmltopdf с нашей вёрсткой и встроенной векторной графикой справляется:
# подписи осей рисунка попадают в текст PDF, значит SVG отрисован, а не пропущен.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/paper_artifact.py
wkhtmltopdf --enable-local-file-access --page-size A4 \
    --margin-top 20mm --margin-bottom 20mm --margin-left 20mm --margin-right 20mm \
    --encoding utf-8 --quiet \
    docs/paper/artifact.html docs/paper/article_gip.pdf
python3 - <<'PY'
import re
d = open("docs/paper/article_gip.pdf", "rb").read()
pages = len(re.findall(rb"/Type\s*/Page[^s]", d))
print("готово: docs/paper/article_gip.pdf — {} стр., {} КБ".format(pages, len(d) // 1024))
PY
