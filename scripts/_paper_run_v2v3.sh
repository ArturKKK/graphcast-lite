#!/usr/bin/env bash
# Сравнение глобальных моделей v2 (19 каналов) и v3 (33 канала) на одном тестовом окне.
# Цель — справочная: подтвердить, что базой для 33-канальной региональной модели взята
# более сильная глобальная модель. НЕ изолирует эффект уровней давления: v3 отличается
# от v2 также curriculum (до 8 шагов против 4), остаточной формулировкой, весами каналов
# и инъекцией шума. Поэтому сопоставление — только по ОБЩИМ 19 полям.
#
# Бюджет диска (лимит платформы ~240 ГБ): глобальный 19f 82 + global_extra 43 +
# 33f-датасет ~64 = ~189 ГБ. Российские датасеты и merge на этой VM держать нельзя.
#
# Запуск: nohup setsid bash scripts/_paper_run_v2v3.sh </dev/null >/dev/null 2>&1 &
# Лог:    /workdir/paper_results/v2v3_master.log
set -uo pipefail
REPO=/workdir/graphcast-lite
VENV=/data/venvs/graphcast
DATA=/data/datasets
BASE=$DATA/wb2_512x256_19f_ar
GEXTRA=$DATA/global_512x256_extra_2010-2021_07deg
D33=$DATA/wb2_512x256_33f_v3
OUT=/workdir/paper_results
NMAX=200          # глобальная сетка 512x256 тяжелее региональной; 200 сроков достаточно

mkdir -p "$OUT"
MASTER="$OUT/v2v3_master.log"
exec >>"$MASTER" 2>&1
cd "$REPO"
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== v2 vs v3 START (commit $GIT_COMMIT) ==="
log "disk: $(du -sh $DATA | cut -f1)"

# ---------- 1. venv ----------
if [[ ! -x "$VENV/bin/python" ]] || ! "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  log "venv/torch отсутствует — ставлю"
  rm -rf "$VENV"
  CONDA_PY=$(ls /home/mlcore/conda/bin/python3* 2>/dev/null | head -1)
  { [[ -n "$CONDA_PY" ]] && "$CONDA_PY" -m venv "$VENV"; } || python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip wheel setuptools
  "$VENV/bin/pip" install -q -r requirements.txt \
    || "$VENV/bin/pip" install -q -r requirements.txt \
         --extra-index-url https://artifactory.tcsbank.ru/artifactory/api/pypi/python-all/simple
fi
source "$VENV/bin/activate"
export PYTHONPATH="$REPO"
log "torch=$(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1)"

# ---------- 2. чекпойнты ----------
if [[ ! -f experiments/wb2_512x256_33f_ar_v3/best_model.pth ]]; then
  CK=$(ls "$DATA"/paper_ckpts.tar.zst 2>/dev/null | head -1)
  [[ -n "$CK" ]] && { log "распаковываю чекпойнты"; tar --use-compress-program=unzstd -xf "$CK" -C "$REPO"; }
fi
for c in experiments/wb2_512x256_19f_ar_v2/best_model.pth experiments/wb2_512x256_33f_ar_v3/best_model.pth; do
  [[ -f "$c" ]] || { log "FATAL: нет $c"; exit 2; }
  log "ckpt $(md5sum "$c" | cut -d' ' -f1)  $c"
done

# ---------- 3. глобальный 19f ----------
if [[ ! -f "$BASE/data.npy" ]]; then
  log "распаковываю глобальный 19f"
  mkdir -p "$BASE"
  command -v zstd >/dev/null 2>&1 || apt-get install -y -q zstd >/dev/null 2>&1 || true
  ARC=$(ls "$DATA"/dataset_512x256.tar.zst 2>/dev/null | head -1)
  [[ -z "$ARC" ]] && { log "FATAL: нет архива dataset_512x256.tar.zst"; exit 3; }
  tar --use-compress-program=unzstd -xf "$ARC" -C "$BASE" --strip-components=1
  find "$BASE" -name "._*" -delete 2>/dev/null || true
  found=$(find "$BASE" -maxdepth 4 -name data.npy -type f | head -1)
  [[ -n "$found" && "$(dirname "$found")" != "$BASE" ]] && mv "$(dirname "$found")"/* "$BASE"/ 2>/dev/null
  [[ -f "$BASE/data.npy" ]] || { log "FATAL: data.npy не найден"; exit 3; }
  rm -f "$ARC"          # сразу, иначе лимит диска
  log "глобальный распакован ($(du -sh "$BASE" | cut -f1)), архив удалён; disk=$(du -sh $DATA|cut -f1)"
fi

# ---------- 4. сборка 33-канального глобального датасета ----------
if [[ ! -f "$D33/data_extra.npy" ]]; then
  BL="$OUT/v3_build.log"
  CMD="python -u scripts/build_v3_extra_with_time.py --base-dir $BASE --extra-dir $GEXTRA --out-dir $D33"
  { echo "### PROVENANCE ###"; echo "# started: $(date -Iseconds) | commit $GIT_COMMIT"; echo "# COMMAND: $CMD"; echo; } > "$BL"
  log "собираю 33f глобальный датасет (~64 ГБ, часы) → $BL"
  eval "$CMD" >> "$BL" 2>&1
  [[ -f "$D33/data_extra.npy" ]] || { log "FATAL: сборка 33f не удалась, см. $BL"; exit 4; }
  log "33f готов ($(du -sh "$D33" | cut -f1)); disk=$(du -sh $DATA|cut -f1)"
fi

# ---------- 5. прогоны ----------
run() {  # run <tag> <exp> <data> <extra...>
  local tag="$1" exp="$2" data="$3"; shift 3
  local lf="$OUT/${tag}.log" npz="$OUT/${tag}_samples.npz"
  local cmd="python -u scripts/predict.py experiments/$exp --data-dir $data --split test_only --ar-steps 4 --max-samples $NMAX --per-channel --no-save --save-sample-metrics $npz $*"
  { echo "### PROVENANCE ###############################################"
    echo "# tag: $tag | started: $(date -Iseconds) | host: $(hostname)"
    echo "# git commit: $GIT_COMMIT | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"
    echo "# dataset: $data"
    echo "#   info: $(tr -d '\n ' < "$data/dataset_info.json" 2>/dev/null | cut -c1-300)"
    echo "# experiment: experiments/$exp (config md5 $(md5sum experiments/$exp/config.json|cut -d' ' -f1))"
    echo "#   ckpt: $(md5sum experiments/$exp/best_model.pth|cut -d' ' -f1)"
    echo "# COMMAND:"; echo "#   $cmd"
    echo "##############################################################"; echo; } > "$lf"
  log "START $tag"
  eval "$cmd" >> "$lf" 2>&1
  local rc=$?
  echo -e "\n### finished: $(date -Iseconds), exit=$rc ###" >> "$lf"
  log "DONE  $tag rc=$rc | $(grep -oE 'Skill: [0-9.]+%' "$lf"|tail -1) | $(grep -E '^\s+t2m' "$lf"|tail -1|tr -s ' '|cut -c1-60)"
}

run v2_global wb2_512x256_19f_ar_v2 "$BASE"
run v3_global wb2_512x256_33f_ar_v3 "$D33"

log "=== ALL DONE ==="
grep -E "DONE " "$MASTER" | tail -5
