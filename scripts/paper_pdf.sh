#!/usr/bin/env bash
# Собирает PDF рукописи: markdown -> страница -> печать в A4.
#
# Движок — weasyprint. Прежде был wkhtmltopdf, и он врал: набирал с шагом
# строки 13,8 pt вместо требуемых журналом 18,0 (кегль 12 × интервал 1,5),
# то есть втискивал на полосу 50 строк вместо 40. Объём рукописи из-за этого
# занижался на четверть — 23 полосы вместо 29. Измерено 19.09.2026 по
# координатам строк в обоих PDF.
#
# weasyprint читает кодировку из <meta charset>, которого в artifact.html нет:
# тот файл — фрагмент под внешнюю обёртку. Поэтому для печати собирается
# отдельная полная страница.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

python3 scripts/paper_artifact.py "$TMP/standalone.html"

python3 - "$TMP/standalone.html" docs/paper/article_gip.pdf <<'PY'
import sys, warnings
warnings.filterwarnings("ignore")
import weasyprint
src, dst = sys.argv[1], sys.argv[2]
doc = weasyprint.HTML(filename=src).render()
doc.write_pdf(dst)
print(f"готово: {dst} — {len(doc.pages)} стр.")
PY

python3 - <<'PY'
import re, statistics, subprocess
out = subprocess.run(["pdftotext", "-bbox", "-f", "4", "-l", "4",
                      "docs/paper/article_gip.pdf", "-"],
                     capture_output=True, text=True).stdout
ys = sorted({round(float(m.group(1)), 1) for m in re.finditer(r'yMin="([\d.]+)"', out)})
gaps = [b - a for a, b in zip(ys, ys[1:]) if 5 < b - a < 40]
if gaps:
    step = statistics.median(gaps)
    ok = "" if abs(step - 18.0) < 0.6 else "  ВНИМАНИЕ: не соответствует требованию!"
    print(f"шаг строки {step:.1f} pt (журнал требует 18,0 = кегль 12 × интервал 1,5){ok}")
PY
