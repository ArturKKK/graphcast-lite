# Команды для VM: расчёты под статью (M1–M5) — ДВЕ VM (H100)

Раскладка: **VM-A** = «ядро» (все AR-4: M2, M3, M5) ≈ 11–15 ч GPU; **VM-B** = «длинные горизонты» (M1 AR-28, M4) ≈ 9–13 ч GPU. Зависимостей между VM нет.

Всё через MLC CLI. Длинные команды — ТОЛЬКО в tmux (mlc exec рвёт длинные bash -c).
`--oi-corr-len` В МЕТРАХ (100 км = 100000). Residual: `multires_merge_freeze6_v2` — БЕЗ `--no-residual`; `multires_nores_*` — С `--no-residual` (иначе AR взрывается).
⚠️ После фиксов predict.py residual-модели поддержаны в усвоении (пошаговый хук в AR-цикле) + появился `--oi-first-k`. Обязателен `git pull` на VM до запусков.

## Этап 0. Подготовка (P0) — ОДИНАКОВО НА ОБЕИХ VM

```bash
mlc job ls | grep -iE "running|graphcast"
export VMA=<имя-первой>   # ядро
export VMB=<имя-второй>   # длинные горизонты

# 0.1. Код (на каждой VM)
mlc job exec $VMA -- bash -lc 'cd /workdir && { cd graphcast-lite && git fetch && git reset --hard origin/main-arthur || git clone -b main-arthur https://github.com/ArturKKK/graphcast-lite.git; } && cd /workdir/graphcast-lite && git log --oneline -1'
# (то же для $VMB; если git недоступен — файлы через base64, спроси Клода)

# 0.2. P0 одним скриптом: venv + распаковка global/Krsk + сборка merge (~1-2 ч)
mlc job exec $VMA -- bash -lc 'cd /workdir/graphcast-lite && setsid nohup bash scripts/_paper_setup_vm.sh </dev/null >/dev/null 2>&1 & echo launched'
mlc job exec $VMB -- bash -lc 'cd /workdir/graphcast-lite && setsid nohup bash scripts/_paper_setup_vm.sh </dev/null >/dev/null 2>&1 & echo launched'
# прогресс:
mlc job exec $VMA -- tail -15 /data/logs/paper_setup.log

# 0.3. ЧЕКПОЙНТЫ — переносит ПОЛЬЗОВАТЕЛЬ своей программой (в git их нет, кроме v2):
#   На ОБЕ VM:  experiments/multires_merge_freeze6_v2/best_model.pth (28 МБ)
#   На VM-A:    experiments/multires_nores_freeze6/{best_model.pth (28М), checkpoint.pth (77М)}
#               experiments/multires_nores_nofreeze/{best_model.pth (28М), checkpoint.pth (77М)}
#   На VM-B:    experiments/multires_nores_freeze6/best_model.pth (28М)
#               experiments/region_krsk_cds_19f/'best_model (18).pth' (4.3М)
#   (wb2_512x256_19f_ar_v2/best_model.pth уже в git — приедет с clone)
```

## Этап 1 (M2). Канонические прогоны ядра — полный test, с сохранением предсказаний

~1–2 ч A100 на модель. Сначала smoke-тест на 5 сэмплах!

```bash
# 1.0. SMOKE (5 сэмплов, убедиться что метрики разумные, ~2 мин)
mlc job exec $VM -- bash -lc 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --ar-steps 4 --max-samples 5 --per-channel --region 50 60 83 98 2>&1 | tail -20'

# 1.1. Флагман merge_freeze6_v2 (residual=true → БЕЗ --no-residual), полный тест
mlc job exec $VM -- tmux new-session -d -s m2a 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-samples 2000 --per-channel --region 50 60 83 98 --save /data/paper_results/m2_flagship_preds.pt > /data/paper_results/m2_flagship.log 2>&1'

# 1.2. nores_freeze6 (--no-residual ОБЯЗАТЕЛЕН)
mlc job exec $VM -- tmux new-session -d -s m2b 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_nores_freeze6 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-residual --region 50 60 83 98 --save /data/paper_results/m2_freeze6_preds.pt > /data/paper_results/m2_freeze6.log 2>&1'

# 1.3. nores_nofreeze (--no-residual ОБЯЗАТЕЛЕН)
mlc job exec $VM -- tmux new-session -d -s m2c 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_nores_nofreeze --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-residual --region 50 60 83 98 --save /data/paper_results/m2_nofreeze_preds.pt > /data/paper_results/m2_nofreeze.log 2>&1'

# 1.4. Внутренняя зона (после 1.1-1.3; быстро, без --save)
mlc job exec $VM -- tmux new-session -d -s m2d 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && for exp in multires_merge_freeze6_v2 multires_nores_freeze6 multires_nores_nofreeze; do NR=""; [ "$exp" != "multires_merge_freeze6_v2" ] && NR="--no-residual"; python scripts/predict.py experiments/$exp --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-samples 2000 --per-channel $NR --region 55.5 56.5 92 94 > /data/paper_results/m2_inner_$exp.log 2>&1; done'

# Проверка прогресса любого прогона:
mlc job exec $VM -- tail -15 /data/paper_results/m2_flagship.log
mlc job exec $VM -- tmux list-sessions
```

## Этап 2 (M3). Контроль конфаундера freeze/nofreeze — равные эпохи

⚠️ predict.py --ckpt грузит strict=False: checkpoint.pth в обёртке молча НЕ загрузится. Сначала распаковать:

```bash
# 2.1. Распаковка model_state_dict из checkpoint.pth (обе модели)
mlc job exec $VM -- bash -lc 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python -c "
import torch
for exp in [\"multires_nores_freeze6\",\"multires_nores_nofreeze\"]:
    ck=torch.load(f\"experiments/{exp}/checkpoint.pth\",map_location=\"cpu\")
    sd=ck.get(\"model_state_dict\",ck)
    torch.save(sd,f\"/data/paper_results/{exp}_ep16_unwrapped.pth\")
    print(exp,\"->epoch\",ck.get(\"epoch\",\"?\"),\"keys\",len(sd))
"'

# 2.2. Прогоны на равных эпохах (то же окно, что M2)
mlc job exec $VM -- tmux new-session -d -s m3 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_nores_freeze6 --ckpt /data/paper_results/multires_nores_freeze6_ep16_unwrapped.pth --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-residual --region 50 60 83 98 > /data/paper_results/m3_freeze6_ep16.log 2>&1 && python scripts/predict.py experiments/multires_nores_nofreeze --ckpt /data/paper_results/multires_nores_nofreeze_ep16_unwrapped.pth --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-samples 2000 --per-channel --no-residual --region 50 60 83 98 > /data/paper_results/m3_nofreeze_ep16.log 2>&1'
```

## Этап 3 (M1). Усвоение на длинных горизонтах (+48…+168 ч)

Главный сюжет DA-раздела. ~2–4 ч на конфигурацию. Сначала smoke AR-28 на 3 сэмплах!

```bash
# 3.0. SMOKE AR-28 (проверить, что таргеты на 28 шагов есть в окне)
mlc job exec $VM -- bash -lc 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --ar-steps 28 --max-samples 3 --region 50 60 83 98 2>&1 | tail -15'

# 3.1. БЕЗ усвоения, AR-28, 200 сэмплов
mlc job exec $VM -- tmux new-session -d -s m1a 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-samples 200 --per-channel --region 50 60 83 98 > /data/paper_results/m1_noda_ar28.log 2>&1'

# 3.2. ОИ 10% на каждом шаге, AR-28
mlc job exec $VM -- tmux new-session -d -s m1b 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-samples 200 --per-channel --region 50 60 83 98 --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed 42 --oi-corr-len 100000 --oi-sigma-o 0.5 > /data/paper_results/m1_oi10_ar28.log 2>&1'

# 3.3. ОИ 1% (реалистичная сеть), L=200 км, AR-28
mlc job exec $VM -- tmux new-session -d -s m1c 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-samples 200 --per-channel --region 50 60 83 98 --assim-method oi --obs-sparsity 0.01 --obs-roi-only --obs-seed 42 --oi-corr-len 200000 --oi-sigma-o 0.5 > /data/paper_results/m1_oi1_ar28.log 2>&1'

# 3.3b. ОИ только первые 4 шага (теперь поддержано: --oi-first-k)
mlc job exec $VM -- tmux new-session -d -s m1e 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-samples 200 --per-channel --region 50 60 83 98 --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed 42 --oi-corr-len 100000 --oi-sigma-o 0.5 --oi-first-k 4 > /data/paper_results/m1_oi10_first4_ar28.log 2>&1'
# 3.4. Дипломная модель для преемственности (nores_freeze6, --no-residual): baseline + OI10
mlc job exec $VM -- tmux new-session -d -s m1d 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_nores_freeze6 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-samples 200 --per-channel --no-residual --region 50 60 83 98 > /data/paper_results/m1_nores_noda_ar28.log 2>&1 && python scripts/predict.py experiments/multires_nores_freeze6 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 28 --max-samples 200 --per-channel --no-residual --region 50 60 83 98 --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed 42 --oi-corr-len 100000 --oi-sigma-o 0.5 > /data/paper_results/m1_nores_oi10_ar28.log 2>&1'
```

## Этап 4 (M5). Честный DA-протокол: подбор на val → отчёт на test, 3 сида

```bash
# 4.1. Свип L × σ_o на VAL (12 конфигураций × ~10 мин, N=200)
mlc job exec $VM -- tmux new-session -d -s m5a 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && for L in 50000 100000 150000 200000; do for S in 0.3 0.5 1.0; do python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split val --ar-steps 4 --max-samples 200 --region 50 60 83 98 --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed 42 --oi-corr-len $L --oi-sigma-o $S > /data/paper_results/m5_val_oi_L${L}_s${S}.log 2>&1; done; done'

# 4.2. После 4.1: выбрать лучшую (L*,S*) по val-логам, прогнать на TEST с 3 сидами
# (подставь L* и S* из результатов!)
mlc job exec $VM -- tmux new-session -d -s m5b 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && for SEED in 42 43 44; do python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-samples 2000 --per-channel --region 50 60 83 98 --assim-method oi --obs-sparsity 0.1 --obs-roi-only --obs-seed $SEED --oi-corr-len 100000 --oi-sigma-o 0.5 > /data/paper_results/m5_test_oi_best_seed${SEED}.log 2>&1; done'

# 4.3. TEST baseline (без DA) + ОИ 1% + нуджинг (α-свип)
mlc job exec $VM -- tmux new-session -d -s m5c 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-samples 2000 --per-channel --region 50 60 83 98 > /data/paper_results/m5_test_baseline.log 2>&1 && python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-samples 2000 --per-channel --region 50 60 83 98 --assim-method oi --obs-sparsity 0.01 --obs-roi-only --obs-seed 42 --oi-corr-len 200000 --oi-sigma-o 0.5 > /data/paper_results/m5_test_oi1.log 2>&1 && for A in 0.5 0.7 0.9; do python scripts/predict.py experiments/multires_merge_freeze6_v2 --data-dir /data/datasets/multires_krsk_19f_merge --split test_only --ar-steps 4 --max-samples 2000 --per-channel --region 50 60 83 98 --assim-method nudging --nudging-mode sequential --nudging-alpha $A --obs-sparsity 0.1 --obs-roi-only --obs-seed 42 > /data/paper_results/m5_test_nudge_a${A}.log 2>&1; done'
```

## Этап 5 (M4). Абляция «зачем единый граф»

```bash
# 5.1. Региональная GNN standalone (имя чекпойнта с пробелом — кавычки!)
#      сначала smoke 5 сэмплов: не знаем use_residual этого конфига
mlc job exec $VM -- bash -lc 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/region_krsk_cds_19f --ckpt "experiments/region_krsk_cds_19f/best_model (18).pth" --data-dir /data/datasets/region_krsk_61x41_19f_2010-2020_025deg --ar-steps 4 --max-samples 5 --per-channel 2>&1 | tail -20'
# если метрики разумные — полный:
mlc job exec $VM -- tmux new-session -d -s m4a 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/region_krsk_cds_19f --ckpt "experiments/region_krsk_cds_19f/best_model (18).pth" --data-dir /data/datasets/region_krsk_61x41_19f_2010-2020_025deg --split test_only --ar-steps 4 --max-samples 2000 --per-channel > /data/paper_results/m4_regional_standalone.log 2>&1'

# 5.2. «Глобальная v2 + интерполяция в ROI» — ГОТОВЫЙ scripts/interpolate_to_region.py:
#      шаг 1: глобальный инференс с сохранением (v2 в git, чекпойнт есть)
mlc job exec $VM -- tmux new-session -d -s m4b 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/predict.py experiments/wb2_512x256_19f_ar_v2 --data-dir /data/datasets/wb2_512x256_19f_ar --split test_only --ar-steps 4 --max-samples 2000 --per-channel --save /data/paper_results/m4_global_preds.pt > /data/paper_results/m4_global.log 2>&1'
#      шаг 2 (после шага 1): интерполяция на сетку 0.25° + метрики против реального ERA5
mlc job exec $VM -- tmux new-session -d -s m4c 'cd /workdir/graphcast-lite && source /data/venvs/graphcast/bin/activate && python scripts/interpolate_to_region.py --predictions /data/paper_results/m4_global_preds.pt --global-data /data/datasets/wb2_512x256_19f_ar --region-data /data/datasets/region_krsk_61x41_19f_2010-2020_025deg --per-channel > /data/paper_results/m4_interp.log 2>&1'
#      ⚠️ smoke сначала: сверить, что даты глобального теста и региона совпали (скрипт писан под совмещённые окна)
```

## Этап 6. Забрать результаты домой

```bash
# все логи (маленькие) — через base64
mlc job exec $VM -- bash -lc 'cd /data/paper_results && tar czf /data/paper_logs.tgz *.log && ls -la /data/paper_logs.tgz'
mlc job exec $VM -- bash -lc 'base64 /data/paper_logs.tgz' > /tmp/paper_logs.b64
base64 -d /tmp/paper_logs.b64 > ~/Developer/graphcast-lite/vm_backup/paper_results/paper_logs.tgz
cd ~/Developer/graphcast-lite/vm_backup/paper_results && tar xzf paper_logs.tgz
# .pt-предсказания (гигабайты) остаются на VM — бутстреп-скрипт гоняем там же
```

## Мониторинг

```bash
mlc job exec $VM -- tmux list-sessions
mlc job exec $VM -- bash -lc 'tail -5 /data/paper_results/*.log | head -60'
mlc job exec $VM -- nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```
