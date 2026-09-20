#!/usr/bin/env bash
# Забрать посчитанные прогоны в репозиторий: npz с посрочными метриками и логи.
#
# Зачем в репозиторий. Всё, что лежит в /workdir, живёт до выключения
# виртуалки, а посрочные метрики — единственное, из чего потом считаются
# доверительные интервалы без повторного инференса. Со статьёй 3 это уже
# спасло: интервалы нашлись в docs/paper/runs спустя месяц.
#
# Безопасно запускать при идущем батче: копируются только дописанные файлы,
# счёт не трогается.
#
# Запуск:  bash scripts/_vm_save_runs.sh
#          затем  GIT_ASKPASS= git push origin main-arthur
set -uo pipefail
REPO=/workdir/graphcast-lite
SRC=${SRC:-/workdir/paper_results}
DST=${DST:-docs/paper/runs/acc_lat_clim}
[[ -d "$REPO" ]] && cd "$REPO"
mkdir -p "$DST"

n=0
for f in "$SRC"/w_*_samples.npz "$SRC"/w_*.log "$SRC"/acc_master.log "$SRC"/acc_clim.log; do
  [[ -f "$f" ]] || continue
  # Файл, который прямо сейчас дописывается, берём только когда он закрыт:
  # незавершённый npz — это обрезанный zip, и читаться он не будет.
  if [[ "$f" == *_samples.npz ]] && command -v fuser >/dev/null 2>&1 \
     && fuser "$f" >/dev/null 2>&1; then
    echo "  пропускаю (пишется прямо сейчас): $(basename "$f")"
    continue
  fi
  cp -p "$f" "$DST"/ && n=$((n + 1))
done
echo "скопировано файлов: $n"
du -sh "$DST"

# -f обязателен: в .gitignore стоят *.npz (строка 192) и *.log (строка 62),
# и без него `git add` молча не добавляет ничего, а скрипт рапортует «нового
# нет». Эти файлы — намеренное исключение: посрочные метрики и есть то, из
# чего потом считаются доверительные интервалы, ровно как 1150 таких же
# файлов, уже лежащих в docs/paper/runs.
git add -f "$DST"
if git diff --cached --quiet; then
  echo "нового нет — коммитить нечего"
  exit 0
fi
git -c user.email="${GIT_AUTHOR_EMAIL:-artur.tabakov1@gmail.com}" \
    -c user.name="${GIT_AUTHOR_NAME:-Artur Tabakov}" \
    commit -qm "прогоны с широтным весом и ACC против климатологии"
echo
echo "закоммичено. осталось отправить:"
echo "    GIT_ASKPASS= git push origin main-arthur"
