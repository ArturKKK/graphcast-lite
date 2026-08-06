#!/usr/bin/env bash
# Инвентаризация VM: что за машина, что на дисках, какие датасеты, чекпойнты и
# результаты прогонов уже есть. Ничего не считает и ничего не удаляет — только читает.
#
# Запуск на VM:  bash scripts/_paper_inventory.sh
# Результат:     docs/paper/runs/inventory_<host>.md  (маленький, коммитится как есть)
set -uo pipefail

REPO="${REPO:-/workdir/graphcast-lite}"
cd "$REPO" 2>/dev/null || { echo "нет репозитория в $REPO"; exit 1; }

HOST=$(hostname)
OUT_DIR="docs/paper/runs"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/inventory_${HOST}.md"

# Каталоги, где могут лежать результаты (на разных VM по-разному)
RESULT_DIRS="/workdir/paper_results /data/paper_results /workdir/results /data/results"

{
  echo "# Инвентаризация VM — $HOST"
  echo
  echo "Снято: $(date -Iseconds)"
  echo

  echo "## Машина"
  echo '```'
  echo "hostname: $HOST"
  echo "gpu:      $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'нет nvidia-smi')"
  echo "gpu busy: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head -3 || echo '-')"
  echo "uptime:   $(uptime -p 2>/dev/null)"
  echo '```'
  echo

  echo "## Диски"
  echo '```'
  df -h /data /workdir / 2>/dev/null | sort -u
  echo "--- крупное в /data ---"
  du -sh /data/* 2>/dev/null | sort -rh | head -15
  echo '```'
  echo

  echo "## Репозиторий"
  echo '```'
  echo "commit:  $(git rev-parse --short HEAD 2>/dev/null)"
  echo "branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "subject: $(git log -1 --pretty=%s 2>/dev/null)"
  echo "--- незакоммиченное ---"
  git status --porcelain 2>/dev/null | head -25
  echo '```'
  echo

  echo "## Окружение"
  echo '```'
  for v in /data/venvs/graphcast /workdir/venvs/graphcast; do
    [[ -x "$v/bin/python" ]] && {
      echo "venv:   $v"
      echo "python: $($v/bin/python -V 2>&1)"
      echo "torch:  $($v/bin/python -c 'import torch;print(torch.__version__,"cuda",torch.cuda.is_available())' 2>/dev/null || echo 'torch не импортируется')"
    }
  done
  echo '```'
  echo

  echo "## Датасеты"
  echo
  echo "| датасет | размер | сроков | узлов/сетка | каналов |"
  echo "|---|---:|---:|---|---:|"
  for d in /data/datasets/*/ /workdir/datasets/*/; do
    [[ -d "$d" ]] || continue
    inf=$(ls "$d"dataset_info*.json 2>/dev/null | head -1)
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    if [[ -n "$inf" ]]; then
      read -r nt grid nf < <(python3 -c "
import json,sys
try:
    j=json.load(open('$inf'))
except Exception:
    print('? ? ?'); sys.exit()
g = str(j.get('n_nodes')) if j.get('n_nodes') else f\"{j.get('n_lon','?')}x{j.get('n_lat','?')}\"
print(j.get('n_time','?'), g, j.get('n_feat', j.get('n_feat_extra','?')))
" 2>/dev/null || echo "? ? ?")
      echo "| $(basename "$d") | $sz | $nt | $grid | $nf |"
    else
      echo "| $(basename "$d") | $sz | — | *нет dataset_info* | — |"
    fi
  done
  echo

  echo "## Чекпойнты"
  echo
  echo "| файл | размер | md5 | изменён |"
  echo "|---|---:|---|---|"
  find experiments -name "*.pth" 2>/dev/null | sort | while read -r p; do
    echo "| $p | $(du -h "$p" | cut -f1) | $(md5sum "$p" | cut -c1-12) | $(date -r "$p" '+%d.%m %H:%M') |"
  done
  echo

  echo "## Результаты прогонов"
  for R in $RESULT_DIRS; do
    [[ -d "$R" ]] || continue
    echo
    echo "### $R"
    echo '```'
    ls -la "$R" 2>/dev/null | head -40
    echo '```'
    # Итоговые строки — самое ценное: сразу видно, что посчиталось
    if compgen -G "$R"/*.log >/dev/null; then
      echo
      echo "Итоговые строки (DONE / skill) из логов:"
      echo '```'
      grep -h -E "DONE |skill=" "$R"/*.log 2>/dev/null | tail -40
      echo '```'
    fi
  done
  echo

  echo "## Незавершённые прогоны (tmux)"
  echo '```'
  tmux list-sessions 2>/dev/null || echo "сессий tmux нет"
  echo '```'
} > "$OUT" 2>&1

echo "готово → $REPO/$OUT"
echo
echo "Кратко:"
grep -c "^| " "$OUT" 2>/dev/null | xargs echo "  строк в таблицах:"
du -h "$OUT" | cut -f1 | xargs echo "  размер отчёта:"
