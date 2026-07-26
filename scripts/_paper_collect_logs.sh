#!/usr/bin/env bash
# Собирает ВСЕ логи расчётов статьи + манифест провенанса в один архив.
# Манифест отвечает на вопросы «что запускали, на каком коммите, какими весами».
# Запуск на VM: bash scripts/_paper_collect_logs.sh [имя-архива]
# Результат:    /workdir/paper_logs_<host>_<дата>.tar.zst  (забирается через base64)
set -uo pipefail
REPO=/workdir/graphcast-lite
OUT=/workdir/paper_results
HOST=$(hostname | sed 's/.*graphcast/graphcast/;s/-task-0//')
STAMP=$(date +%Y%m%d_%H%M)
ARC="${1:-/workdir/paper_logs_${HOST}_${STAMP}.tar.zst}"
MAN="$OUT/MANIFEST.md"

cd "$REPO"
mkdir -p "$OUT"

{
  echo "# Манифест прогонов — $HOST, $(date -Iseconds)"
  echo
  echo "## Репозиторий"
  echo '```'
  echo "commit: $(git rev-parse HEAD 2>/dev/null)"
  echo "short:  $(git rev-parse --short HEAD 2>/dev/null)"
  echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "subject: $(git log -1 --pretty=%s 2>/dev/null)"
  echo "date:    $(git log -1 --pretty=%ci 2>/dev/null)"
  echo "--- uncommitted (если есть) ---"
  git status --porcelain 2>/dev/null | head -20
  echo '```'
  echo
  echo "## Окружение"
  echo '```'
  echo "gpu:    $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
  echo "torch:  $(/data/venvs/graphcast/bin/python -c 'import torch;print(torch.__version__, "cuda", torch.cuda.is_available())' 2>/dev/null)"
  echo "python: $(/data/venvs/graphcast/bin/python -V 2>&1)"
  echo '```'
  echo
  echo "## Датасеты"
  for d in /data/datasets/*/; do
    inf=$(ls "$d"dataset_info*.json 2>/dev/null | head -1)
    [[ -z "$inf" ]] && continue
    echo "### $(basename "$d")  ($(du -sh "$d" 2>/dev/null | cut -f1))"
    echo '```json'
    tr -d '\n' < "$inf" | cut -c1-500; echo
    echo '```'
  done
  echo
  echo "## Чекпойнты (md5)"
  echo '```'
  find experiments -name "*.pth" -printf "%10s  %p\n" 2>/dev/null | sort -k2 | while read -r sz p; do
    echo "$(md5sum "$p" | cut -d' ' -f1)  $(du -h "$p" | cut -f1)  $p"
  done
  echo '```'
  echo
  echo "## Логи прогонов"
  echo "| файл | размер | изменён | команда (из PROVENANCE) |"
  echo "|---|---|---|---|"
  for f in "$OUT"/*.log; do
    [[ -f "$f" ]] || continue
    cmd=$(grep -A1 "^# COMMAND:" "$f" 2>/dev/null | tail -1 | sed 's/^#\s*//' | cut -c1-160)
    [[ -z "$cmd" ]] && cmd="(без header — см. scripts/_paper_run_*.sh на коммите выше)"
    echo "| $(basename "$f") | $(du -h "$f" | cut -f1) | $(date -r "$f" +%H:%M) | \`$cmd\` |"
  done
  echo
  echo "## Итоговые строки (DONE) из мастер-логов"
  echo '```'
  grep -hE "DONE|ALL DONE" "$OUT"/*master*.log 2>/dev/null | tail -40
  echo '```'
} > "$MAN"

# в архив: логи, манифест, компактные npz с per-sample метриками (если есть)
tar -C "$OUT" -cf - . 2>/dev/null | zstd -3 -o "$ARC" -f >/dev/null 2>&1
echo "MANIFEST: $MAN"
echo "ARCHIVE:  $ARC  ($(du -h "$ARC" | cut -f1))"
echo "MD5:      $(md5sum "$ARC" | cut -d' ' -f1)"
