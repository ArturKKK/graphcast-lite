#!/usr/bin/env bash
# Декодировщик с признаками рёбер поверх основной модели chw (24.09.2026).
#
# Зачем. Разложение ошибки по масштабам (docs/results/error_scales_2026-09-24.md)
# показало отпечаток треугольников меша в прогнозе: GCN-декодировщик даёт трём
# вершинам треугольника одинаковый вес. Новый декодировщик (как в GraphCast)
# учитывает смещение узла сетки относительно каждой вершины.
#
# Как. Кодировщик, процессор и MLP декодировщика берём из chw (эпоха 7), новым
# учится только слой сообщений декодировщика. Его выход инициализирован нулём,
# так что старт равен инерционному прогнозу. Первую эпоху процессор заморожен.
# Дальше 8 эпох как у chw. Всего от krsk33f выходит 8 + 8 = 16 эпох, столько же,
# сколько у варианта long: с ним и сравниваем при равном бюджете.
# ~26 ч обучения (слой дороже GCN примерно на 20 %) + ~2 ч оценки.
#
# Запуск:  bash scripts/_vm_batch_decoder.sh
#          bash scripts/_vm_batch_decoder.sh ema   — встать в очередь: дождаться,
#          пока на этой машине закончится батч ema (обучение и его оценка), и
#          сразу стартовать. Чтобы не вставать ночью ради запуска.
#          bash scripts/_vm_batch_decoder.sh +long — то же, но поверх модели
#          long (16 эпох) вместо chw: декодировщик на лучшей базе. Опыт
#          dec_long, лог improve_dec_long_master.log. Аргументы сочетаются:
#          «ema +long» — ждать ema и потом стартовать от long.
#          bash scripts/_vm_batch_decoder.sh dec +enc — дождаться батча dec и
#          поставить опыт dec_enc: вместе с декодировщиком заменить и
#          кодировщик (GCN → сообщения с признаками рёбер, как в GraphCast).
#          База та же chw и те же 8 эпох, что у dec: чистое сравнение
#          «только декодировщик» против «оба блока» при равном бюджете. ПРОВАЛИЛСЯ
#          (val 0,00737 против 0,00341 на 3-й эпохе): процессор разом терял вход.
#          bash scripts/_vm_batch_decoder.sh +encres — поверх готовой модели dec
#          (с «+long» — поверх dec_long) добавить к GCN-кодировщику поправку
#          сообщениями с признаками рёбер, стартующую с нуля. Модель начинает
#          ровно с dec и хуже неё стать не может. Опыт dec_encres
#          (dec_long_encres). На v4 в очередь за dec_long: «dec_long +long +encres».
# Лог:     /workdir/paper_results/improve_dec_master.log
set -uo pipefail
V=dec
AFTER=""
BASE=chw
ENC=0
ENCRES=0
for a in "$@"; do
  case "$a" in
    ema|long|dec|dec_long) AFTER=$a ;;
    +long)    BASE=long ;;
    +enc)     ENC=1 ;;
    +encres)  ENCRES=1 ;;
    *) echo "непонятный аргумент «$a»: ждать — ema, long, dec, dec_long; база — +long; кодировщик — +enc, +encres"; exit 1 ;;
  esac
done
[[ "$BASE" == "long" ]] && V=dec_long
[[ "$ENC" == "1" && "$ENCRES" == "1" ]] && { echo "+enc и +encres вместе нельзя"; exit 1; }
[[ "$ENC" == "1" ]] && V=${V}_enc
DECSRC=multires_krsk_33f_chw_${V}      # готовая модель с декодировщиком — база для +encres
[[ "$ENCRES" == "1" ]] && V=${V}_encres

if [[ "${DAEMONIZED:-}" != "1" ]]; then
  mkdir -p /workdir/paper_results
  DAEMONIZED=1 setsid nohup bash "$0" "$@" </dev/null >/dev/null 2>&1 &
  echo "запущено в фоне. следить:  tail -f /workdir/paper_results/improve_${V}_master.log"
  exit 0
fi

REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
OUT=/workdir/paper_results
HEAVY=/data/paper_heavy
D33=/data/datasets/multires_krsk_33f
ROI="50 60 83 98"
CLIM=$REPO/docs/paper/runs/clim_wb2_nodes.npz
SRC=multires_krsk_33f_chw
[[ "$BASE" == "long" ]] && SRC=multires_krsk_33f_chw_long
[[ "$ENCRES" == "1" ]] && SRC=$DECSRC
EXP=multires_krsk_33f_chw_${V}

mkdir -p "$OUT" "$HEAVY"
MASTER="$OUT/improve_${V}_master.log"
exec 9>"$OUT/.improve_${V}.lock"
flock -n 9 || { echo "[$(date '+%d.%m %H:%M:%S')] уже идёт" >> "$MASTER"; exit 0; }
exec >>"$MASTER" 2>&1
cd "$REPO" || exit 1
log() { echo "[$(date '+%d.%m %H:%M:%S')] $*"; }
log "=== УЛУЧШЕНИЕ: $V ($(git rev-parse --short HEAD)) ==="

# Очередь. Батч improve держит свою блокировку до самого конца, и её
# наследуют его python-процессы, так что блокировка освобождается, только
# когда закончены и обучение, и оценка.
if [[ -n "$AFTER" ]]; then
  log "жду окончания батча $AFTER (блокировка $OUT/.improve_${AFTER}.lock)"
  exec 8>>"$OUT/.improve_${AFTER}.lock"
  flock 8
  exec 8>&-
  log "батч $AFTER закончился — начинаю"
  sleep 60     # дать карте освободить память
fi

BUSY=$(pgrep -af "^python.*(src\.main|scripts/predict\.py)" | head -1)
[[ -n "$BUSY" ]] && { log "карта занята: $BUSY — стоп"; exit 1; }

# ---------- окружение и датасет ----------
if [[ ! -x "$VENV/bin/python" || ! -f "$D33/data_extra.npy" ]]; then
  log "нет окружения или датасета — восстанавливаю (~40 мин), лог /workdir/paper_logs/restore33f.log"
  DAEMONIZED=1 FREE=1 bash scripts/_vm_restore33f.sh
  log "восстановление rc=$?"
fi
[[ -f "$D33/data_extra.npy" ]] || { log "датасет так и не собрался — стоп"; exit 1; }
source "$VENV/bin/activate" || { log "нет venv — стоп"; exit 1; }
export PYTHONPATH="$REPO"

# ---------- состояния моделей ----------
# Стартуем с chw: её кодировщик и процессор уже дообучены под регион.
extract() {   # checkpoint.pth -> голый state_dict
  python - "$1" "$2" <<'PY'
import sys, pathlib, torch
src, dst = pathlib.Path(sys.argv[1]), sys.argv[2]
if not src.exists():
    print(f"[prep] нет {src}"); raise SystemExit(1)
ck = torch.load(src, map_location="cpu")
torch.save(ck.get("model_state_dict", ck), dst)
print(f"[prep] {src.parent.name}: эпоха {ck.get('epoch','?')} -> {dst}")
PY
}
BASE_ST=$HEAVY/${SRC}_start.pth
[[ "$SRC" == "multires_krsk_33f_chw" ]] && BASE_ST=$HEAVY/krsk33f_chw_last.pth
[[ -f "$BASE_ST" ]] || extract "experiments/$SRC/checkpoint.pth" "$BASE_ST" \
  || { log "нет состояния $SRC — стоп"; exit 1; }
START=$BASE_ST

# ---------- оценка ----------
run() {   # run <тег> <опыт> <чекпойнт> [доп. ключи]
  local tag="$1" exp="$2" ck="$3"; shift 3
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  [[ -f "$npz" ]] && { log "SKIP $tag (уже посчитан)"; return 0; }
  log "START $tag"
  python -u scripts/predict.py "experiments/$exp" --data-dir "$D33" \
      --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-save \
      --region $ROI --lat-weight --ckpt "$ck" \
      --save-sample-metrics "$npz" --save-region-errors "$OUT/${tag}_errors.npz" "$@" \
      > "$lf" 2>&1
  local rc=$? t2
  t2=$(grep -E "^\s+t2m" "$lf" | tail -1 | tr -s ' ' | cut -c1-58)
  log "DONE  $tag rc=$rc | $t2"
}

# ---------- конфиг ----------
mkdir -p "experiments/$EXP"
python - "experiments/$SRC/config.json" "experiments/$EXP/config.json" "$ENC" "$ENCRES" <<'PY'
import json, sys
src, dst, enc, encres = sys.argv[1:5]
c = json.load(open(src))
if encres == "1":
    g = c["pipeline"]["encoder"]["gcn"]
    assert g["layer_type"] == "conv_gcn", g
    g["edge_refine"] = True
    g["edge_feature_dim"] = 4
if enc == "1":
    c["pipeline"]["encoder"]["gcn"] = {
        "layer_type": "interaction_net_encoder", "hidden_dims": [256],
        "output_dim": c["pipeline"]["encoder"]["gcn"]["output_dim"],
        "activation": "swish", "edge_feature_dim": 4, "use_layer_norm": True}
c["pipeline"]["decoder"]["gcn"] = {
    "layer_type": "interaction_net_decoder", "hidden_dims": [128],
    "output_dim": c["pipeline"]["decoder"]["gcn"]["output_dim"],
    "activation": "swish", "edge_feature_dim": 4, "use_layer_norm": True}
c["num_epochs"] = 8                     # у long в конфиге 16; дообучение всегда 8
c["freeze_processor_epochs"] = 1        # первую эпоху учится новый слой, процессор не трогаем
c["finetune_processor_lr_factor"] = 1.0
c["early_stopping_patience"] = 100      # косинус до нуля: останавливаться рано незачем
c["_comment"] = "chw_dec: декодировщик с признаками рёбер, см. scripts/_vm_batch_decoder.sh"
json.dump(c, open(dst, "w"), indent=2, ensure_ascii=False)
print(f"[prep] {dst}: эпох {c['num_epochs']}, кодировщик {c['pipeline']['encoder']['gcn']['layer_type']}"
      f"{' + поправка' if c['pipeline']['encoder']['gcn'].get('edge_refine') else ''}, "
      f"декодировщик {c['pipeline']['decoder']['gcn']['layer_type']}, "
      f"заморозка процессора {c['freeze_processor_epochs']}, темп {c['learning_rate']} {c.get('lr_schedule')}")
PY
python - "experiments/$EXP/config.json" <<'PY' || { log "конфиг не проходит схему — стоп"; exit 1; }
import json, sys
from src.config import ExperimentConfig
ExperimentConfig(**json.load(open(sys.argv[1])))
PY

# ---------- обучение ----------
RESUME=""
[[ -f "experiments/$EXP/checkpoint.pth" ]] && { RESUME="--resume"; log "нашёлся чекпойнт — продолжаю"; }
log "START обучение $EXP $RESUME"
python -u -m src.main "experiments/$EXP" --pretrained "$START" $RESUME \
    >> "$OUT/improve_${V}_train.log" 2>&1
log "DONE  обучение rc=$?"
tail -12 "experiments/$EXP/training_log.txt" 2>/dev/null

# ---------- оценка результата ----------
FIN=$HEAVY/${EXP}_last.pth
extract "experiments/$EXP/checkpoint.pth" "$FIN" || { log "нет чекпойнта — оценки не будет"; exit 1; }
run t_${V}_roi "$EXP" "$FIN" --climatology "$CLIM"

log "=== ВСЁ ==="
grep -E "DONE  " "$MASTER" | tail -4
