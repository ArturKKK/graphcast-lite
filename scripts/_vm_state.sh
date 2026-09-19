#!/usr/bin/env bash
# Что есть на виртуалке: диски, датасеты, окружение, чекпойнты, результаты.
#
# Зачем скриптом, а не одной командой. Длинная строка в терминале пользователя
# переносится, и хвост уходит отдельной командой — 19.09.2026 так развалилась
# проверка состояния, и половина ответов потерялась. Логика живёт здесь, а
# вставлять надо короткое:  bash scripts/_vm_state.sh
#
# НИЧЕГО НЕ ПИШЕТ на диск: в /workdir квота 8 ГБ, при превышении job убивают.
set -uo pipefail
DATA=/data
WD=/workdir
REPO=/workdir/graphcast-lite
say() { printf '\n== %s\n' "$*"; }

say "диски"
df -h / 2>/dev/null | tail -1
for d in "$WD" "$DATA"; do
  [[ -d "$d" ]] || { echo "  $d — НЕТ"; continue; }
  # /workdir: квота 8 ГБ, выход за неё гасит виртуалку. /data: не учитывается,
  # но стирается при рестарте.
  echo "  $d: $(du -sh "$d" 2>/dev/null | cut -f1)"
done
[[ -d "$WD" ]] && { echo "  крупное в $WD:"; du -sh "$WD"/* 2>/dev/null | sort -rh | head -6 | sed 's/^/     /'; }

say "окружение"
V=$DATA/venvs/graphcast
if [[ -x "$V/bin/python" ]]; then
  echo "  venv: есть — $("$V/bin/python" -c 'import torch;print("torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>&1 | head -1)"
else
  echo "  venv: НЕТ ($V) — ставится scripts/_paper_setup_vm.sh с VENV_ONLY=1"
fi

say "датасеты в $DATA/datasets"
if [[ -d "$DATA/datasets" ]]; then
  du -sh "$DATA"/datasets/* 2>/dev/null | sort -rh | sed 's/^/  /' || echo "  пусто"
else
  echo "  каталога нет — /data стёрся"
fi

say "архивы для восстановления (ждём их из S3)"
ls -la "$DATA"/*.tar.zst "$DATA"/*.tar.gz "$DATA"/*.tar 2>/dev/null | sed 's/^/  /' || echo "  архивов нет"

say "репозиторий"
if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" log --oneline -1 2>/dev/null | sed 's/^/  HEAD: /'
  echo "  ветка: $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "  незакоммиченного: $(git -C "$REPO" status --short 2>/dev/null | wc -l) файлов"
else
  echo "  $REPO — не репозиторий"
fi

say "чекпойнты (*.pth)"
find "$REPO/experiments" "$DATA" -maxdepth 4 -name '*.pth' -printf '  %10s Б  %p\n' 2>/dev/null | sort -k2 -rn | head -25
echo "  ---- нужные поимённо:"
for c in \
  "$REPO/experiments/multires_merge_freeze6_v2/best_model.pth" \
  "$REPO/experiments/multires_nores_freeze6/best_model.pth" \
  "$REPO/experiments/multires_nores_nofreeze/best_model.pth" \
  "$WD/paper_results/krsk33f_last_epoch.pth"
do
  [[ -f "$c" ]] && echo "     ЕСТЬ  $c" || echo "     нет   $c"
done

say "результаты в $WD/paper_results"
if [[ -d "$WD/paper_results" ]]; then
  echo "  всего файлов: $(ls -1 "$WD/paper_results" 2>/dev/null | wc -l), объём $(du -sh "$WD/paper_results" 2>/dev/null | cut -f1)"
  echo "  npz:"
  ls -la "$WD"/paper_results/*.npz 2>/dev/null | sed 's/^/     /' | head -20 || echo "     нет"
  # clim_krsk.npz — климатология по всем каналам, считалась 12 мин 15.08.2026.
  # Если уцелела, агрегат относительно климатологии считается без прогонов.
  [[ -f "$WD/paper_results/clim_krsk.npz" ]] \
    && echo "  clim_krsk.npz: ЕСТЬ — пересчитывать не надо" \
    || echo "  clim_krsk.npz: нет — 12 мин на процессоре"
else
  echo "  каталога нет"
fi

say "видеокарта"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  || nvidia-smi 2>/dev/null | sed -n '9,12p' \
  || echo "  nvidia-smi недоступен"

printf '\n== конец\n'
