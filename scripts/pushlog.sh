#!/usr/bin/env bash
# Забрать логи текущих обучений в репозиторий и запушить.
#
# Существует потому, что длинные команды не переживают вставку в терминал:
# перенос строки рвёт их на куски, и куски выполняются как отдельные команды.
# Здесь всё уже собрано, на виртуалке остаётся набрать одно короткое слово.
#
# Запуск: bash scripts/pushlog.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# Каталог свой у каждой виртуалки. Иначе две машины пишут одни и те же имена
# файлов, затирают друг друга и ловят конфликт на rebase.
TAG=$(hostname | grep -oE 'graphcast-v[0-9]+-[0-9]+' | head -1)
[[ -z "$TAG" ]] && TAG=$(hostname | tr -c 'a-zA-Z0-9-' '-' | cut -c1-40)
DEST="docs/paper/runs/$TAG"
mkdir -p "$DEST"
echo "каталог прогонов: $DEST"

# Логи обучений: training_log.txt из каждого каталога эксперимента, где он
# обновлялся за последние сутки, плюс master-логи запускающих скриптов.
n=0
for f in experiments/*/training_log.txt; do
  [[ -f "$f" ]] || continue
  exp=$(basename "$(dirname "$f")")
  cp "$f" "$DEST/${exp}_training_log.txt" && n=$((n+1))
done
for f in /workdir/paper_results/*.log; do
  [[ -f "$f" ]] && cp "$f" "$DEST/$(basename "$f")" && n=$((n+1))
done
# Поканальные метрики по каждому сроку — из них считаются доверительные
# интервалы блочным бутстрепом. Файлы мелкие, тянем сразу, чтобы не ездить дважды.
for f in /workdir/paper_results/*_samples.npz; do
  [[ -f "$f" ]] && cp "$f" "$DEST/$(basename "$f")" && n=$((n+1))
done
echo "скопировано файлов: $n"

# Короткая сводка на экран, чтобы не гонять отдельную команду
for f in "$DEST"/*_training_log.txt; do
  [[ -f "$f" ]] || continue
  echo "--- $(basename "$f")"
  tail -3 "$f"
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

# Файлы внутри каталога попадают под правила .gitignore (*.log и прочее), а
# `git add -f <каталог>` их не пробивает — 13.08.2026 из-за этого скрипт молча
# написал «нечего коммитить», хотя скопировал 80 файлов. Перечисляем поимённо.
find "$DEST" -type f -print0 | xargs -0 --no-run-if-empty git add -f -- || exit 1
# Отслеживаемые логи в каталогах экспериментов дописываются прямо во время
# обучения. Если их не заигнорить, rebase упирается в незакоммиченные изменения.
git add -f experiments/*/training_log.txt 2>/dev/null
if git diff --cached --quiet; then
  echo "нечего коммитить — логи не изменились"
  exit 0
fi
git commit -qm "логи обучения $(date '+%d.%m %H:%M')" || exit 1
# --autostash: обучение могло дописать лог за те секунды, что шёл коммит
git pull -q --rebase --autostash origin main-arthur || { echo "ОШИБКА pull"; exit 1; }
git push -q origin main-arthur || { echo "ОШИБКА push"; exit 1; }
echo PUSHED
