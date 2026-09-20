#!/usr/bin/env bash
# Упаковка чекпойнтов, на которых стоят числа статей, в один архив для скачивания.
#
# Зачем. В .gitignore стоит `experiments/*/*.pth`, поэтому обученные модели
# существуют ТОЛЬКО на виртуалках. Наборы у машин разные: 19.09.2026 на v3
# нашлись chwb, seed43, roiw100 и merge_freeze6_v2, которых на v4 нет, а на v4 —
# multimesh, roiw30 и region_krsk_cds_19f, которых нет на v3. Погасить машину,
# не забрав её чекпойнты, значит потерять возможность пересчитать таблицы.
#
# Архив кладётся в /data (не в /workdir — там квота 8 ГБ) и его надо СКАЧАТЬ:
# /data стирается при рестарте.
#
# Запуск:  bash scripts/_vm_pack_ckpts.sh          — только показать, что нашлось
#          bash scripts/_vm_pack_ckpts.sh --pack   — собрать архив
set -uo pipefail
REPO=/workdir/graphcast-lite
[[ -d "$REPO" ]] && cd "$REPO"
OUT=${OUT:-/data/ckpts_$(hostname | tr -cd 'a-zA-Z0-9-').tar}

# Опыты, на которых стоят числа статей. Имена — из таблиц:
#   33f            — «33 канала, базовая» (табл. 1)
#   chw, chwb      — взвешивание каналов, основная конфигурация статьи
#   roiw10/30/100  — вес целевой области (табл. 3)
#   seed43         — повторное обучение, межзапусковая изменчивость
#   nores_*        — пара заморозка/без заморозки (табл. 4)
#   merge_freeze6_v2, region_krsk_cds_19f — 19-канальная линия
WANT=(
  multires_krsk_33f multires_krsk_33f_chw multires_krsk_33f_chwb
  multires_krsk_33f_roiw10 multires_krsk_33f_roiw30 multires_krsk_33f_roiw100
  multires_krsk_33f_seed43 multires_nores_freeze6 multires_nores_nofreeze
  multires_merge_freeze6_v2 region_krsk_cds_19f
)

found=(); total=0
for e in "${WANT[@]}"; do
  hit=0
  while IFS= read -r f; do
    [[ -f "$f" ]] || continue
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    total=$((total + sz))
    found+=("$f")
    hit=1
    printf '  %6s МБ  %s\n' "$((sz / 1048576))" "$f"
  done < <(find "experiments/$e" -maxdepth 1 \( -name '*.pth' -o -name 'config.json' \) 2>/dev/null)
  [[ "$hit" == 0 ]] && printf '     ---     experiments/%s — нет на этой машине\n' "$e"
done

echo
echo "итого: ${#found[@]} файлов, $((total / 1048576)) МБ"
[[ ${#found[@]} -eq 0 ]] && exit 0

if [[ "${1:-}" != "--pack" ]]; then
  echo "это был просмотр. собрать:  bash scripts/_vm_pack_ckpts.sh --pack"
  exit 0
fi

# Без сжатия: веса модели — это float, zstd на них выигрывает единицы процентов,
# а время тратит заметное.
printf '%s\n' "${found[@]}" | tar -cf "$OUT" -T -
echo
echo "готово: $OUT  ($(du -h "$OUT" | cut -f1))"
echo "СКАЧАЙ ЕГО: /data стирается при рестарте виртуалки"
