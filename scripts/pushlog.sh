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

# Файлы внутри каталога попадают под правила .gitignore (*.log и прочее).
# Нужны сразу оба флага: -f пробивает игнор, -A забирает весь каталог.
# 13.08.2026 дважды напоролись на молчаливое «нечего коммитить» из-за нехватки
# одного из них, поэтому здесь же печатаем, сколько реально встало в индекс.
git add -Af docs/paper/runs || exit 1
# Отслеживаемые логи в каталогах экспериментов дописываются прямо во время
# обучения. Если их не заигнорить, rebase упирается в незакоммиченные изменения.
git add -f experiments/*/training_log.txt 2>/dev/null
# Итоги оценок — мелкие md/json, но в них весь результат прогона. Под .gitignore
# они не попадают, однако и в индекс сами не встают: pushlog забирал только
# логи, и разбивка по срокам нейронного постпроцессора осталась бы жить лишь на
# виртуалке, то есть до её пересоздания.
git add -f experiments/*/eval_*/*.md experiments/*/eval_*/*.json 2>/dev/null
# Нормировки и учёт прогонов: мелкие json, но по ним восстанавливается состав
# признаков модели, а без него строка в учёте неполна.
git add -f experiments/*/scalers.json experiments/*/station_to_idx.json 2>/dev/null
git add -f docs/postproc_runs.md 2>/dev/null
staged=$(git diff --cached --name-only | wc -l)
echo "в индексе файлов: $staged"
if [[ "$staged" -eq 0 ]]; then
  # Раньше здесь стоял exit — и это съело два прогона: первый запуск коммитил
  # логи, но падал на пуше (не было кредов), а следующие видели «нечего
  # коммитить» и выходили, не доводя коммит до сервера. Теперь выходим только
  # если и пушить нечего.
  if git diff --quiet HEAD @{upstream} 2>/dev/null; then
    echo "нечего коммитить и нечего пушить"
    exit 0
  fi
  echo "новых логов нет, но есть незапушенные коммиты — отправляю"
else
  git commit -qm "логи обучения $(date '+%d.%m %H:%M')" || exit 1
fi
# --autostash: обучение могло дописать лог за те секунды, что шёл коммит
git pull -q --rebase --autostash origin main-arthur || { echo "ОШИБКА pull"; exit 1; }
git push -q origin main-arthur || { echo "ОШИБКА push"; exit 1; }
echo PUSHED
