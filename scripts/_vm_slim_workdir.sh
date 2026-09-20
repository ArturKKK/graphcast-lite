#!/usr/bin/env bash
# Разгрузка /workdir: тяжёлое переносится в /data, где объём не учитывается.
#
# В /workdir квота 8 ГБ, и её превышение гасит job без предупреждения.
# 19.09.2026 на одной из машин было занято 7,8 ГБ из 8.
#
# Ничего не удаляет: переносит и оставляет символическую ссылку, так что пути
# в скриптах продолжают работать. Обратная сторона — /data стирается при
# рестарте, поэтому переносим только то, что восстановимо: содержимое data/,
# которое git не отслеживает (матрица высот, корпуса, выгрузки наблюдений).
#
# Запуск:  bash scripts/_vm_slim_workdir.sh          — только показать
#          bash scripts/_vm_slim_workdir.sh --move   — перенести
#
# KEEP — что НЕ переносить, оставить в /workdir. По умолчанию это базовый
# корпус постобработки: он весит 342 МБ, но собирался 21 час на видеокарте
# (27.08.2026 14:33 -> 28.08 11:27), тогда как производные от него наборы
# (окрестности, лаги, рельеф) пересчитываются из него за минуты.
set -uo pipefail
REPO=/workdir/graphcast-lite
DEST=${DEST:-/data/workdir_offload}
KEEP=${KEEP:-corpus_krsk_2016_2020.parquet}
# На виртуалке это $REPO; вне её работаем в текущем репозитории, чтобы скрипт
# можно было прогнать вхолостую и посмотреть, что он посчитает кандидатами.
[[ -d "$REPO" ]] && cd "$REPO"
git rev-parse --git-dir >/dev/null 2>&1 || { echo "не репозиторий: $PWD"; exit 1; }

candidates() {
  # Порции через -z: в репозитории есть путь с пробелами, и разбор по полям
  # его бы разорвал. Код состояния — первые два символа, путь — с третьего.
  git status --porcelain -z --ignored data/ 2>/dev/null \
    | while IFS= read -r -d '' e; do
        case "${e:0:2}" in '??'|'!!') printf '%s\n' "${e:3}" ;; esac
      done | sed 's#/$##' | sort -u
}

# Внутрь каталога-кандидата спускаемся, если там лежит что-то из KEEP: иначе
# перенос утащил бы вместе с дешёвыми производными и дорогой базовый корпус.
expand() {
  local p
  while IFS= read -r p; do
    if [[ -d "$p" && -n "$KEEP" ]] && find "$p" -maxdepth 1 -name "$KEEP" | grep -q .; then
      find "$p" -maxdepth 1 -mindepth 1 ! -name "$KEEP"
    else
      printf '%s\n' "$p"
    fi
  done < <(candidates)
}

echo "занято в /workdir: $(du -sh /workdir 2>/dev/null | cut -f1) из 8 ГБ"
echo
echo "кандидаты на перенос (в data/, git их не отслеживает):"
found=0
while IFS= read -r p; do
  [[ -e "$p" ]] || continue
  # Уже перенесённое пропускаем: символическая ссылка весит ноль.
  [[ -L "$p" ]] && continue
  sz=$(du -sh "$p" 2>/dev/null | cut -f1)
  echo "    $sz  $p"
  found=1
done < <(expand)
[[ "$found" == 0 ]] && { echo "    нечего переносить"; exit 0; }

if [[ "${1:-}" != "--move" ]]; then
  echo
  echo "это был просмотр. перенести:  bash scripts/_vm_slim_workdir.sh --move"
  exit 0
fi

mkdir -p "$DEST"
while IFS= read -r p; do
  [[ -e "$p" && ! -L "$p" ]] || continue
  tgt="$DEST/$p"
  mkdir -p "$(dirname "$tgt")"
  echo "перенос: $p -> $tgt"
  mv "$p" "$tgt" && ln -s "$tgt" "$p"
done < <(expand)
echo
echo "стало в /workdir: $(du -sh /workdir 2>/dev/null | cut -f1)"
echo "ВНИМАНИЕ: /data стирается при рестарте — перенесённое придётся качать заново"
