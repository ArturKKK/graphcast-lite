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
DEST=docs/paper/runs/vm4_v4global
mkdir -p "$DEST"

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
echo "скопировано файлов: $n"

# Короткая сводка на экран, чтобы не гонять отдельную команду
for f in "$DEST"/*_training_log.txt; do
  [[ -f "$f" ]] || continue
  echo "--- $(basename "$f")"
  tail -3 "$f"
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

git add -f "$DEST" || exit 1
if git diff --cached --quiet; then
  echo "нечего коммитить — логи не изменились"
  exit 0
fi
git commit -qm "логи обучения $(date '+%d.%m %H:%M')" || exit 1
git pull -q --rebase origin main-arthur || { echo "ОШИБКА pull"; exit 1; }
git push -q origin main-arthur || { echo "ОШИБКА push"; exit 1; }
echo PUSHED
