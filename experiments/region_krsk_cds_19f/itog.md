# Региональная модель Красноярск (region_krsk_cds_19f)

## Эксперимент 1: 259 mesh-узлов (mesh_levels=[3,5], hidden=128, 8 MP steps)

Конфиг: hidden=128, lr=3e-4, 50 эпох, max_ar=4, без static_channels

Обучение (17 эпох, лучший результат):
```
Epoch 15 (AR=2) with attention threshold 0.05424
[Epoch 16] train_loss=0.12649  val_loss=0.09687  val_ACC=0.8699
Patience counter is now 4 

Epoch 16 (AR=2) with attention threshold 0.059664
[Epoch 17] train_loss=0.12454  val_loss=0.09524  val_ACC=0.8708
Val loss reduced by 0.00026. Saving best model... 
```

Инференс (predict.py --ar-steps 4 --max-samples 200):
```
============================================================
=== Inference summary (200 samples) ===
Grid: 61x41 (G=2501) | C=19 | AR=4 horizons
Overall: MSE=0.118393 | RMSE=0.344083 | MAE=0.212416
Baseline: RMSE=0.489345 | MAE=0.284239
Skill: 29.69%
ACC: 0.7931 | base: 0.6883 | Δ=+0.1049

Per-horizon:
  +06h: RMSE=0.209395 | base=0.313348 | skill=33.17% | ACC=0.8928 (base 0.8385)
  +12h: RMSE=0.293816 | base=0.454923 | skill=35.41% | ACC=0.8286 (base 0.7072)
  +18h: RMSE=0.377724 | base=0.542019 | skill=30.31% | ACC=0.7574 (base 0.6264)
  +24h: RMSE=0.448021 | base=0.599090 | skill=25.22% | ACC=0.6938 (base 0.5810)

Per-horizon per-channel RMSE (physical units):
         var   unit      +6h     +12h     +18h     +24h
         t2m      K    2.66°C    3.14°C    3.69°C    4.33°C
```

---

## Эксперимент 2: 1029 mesh-узлов (mesh_levels=[3,6], hidden=128, 8 MP steps)

Конфиг: hidden=128, lr=3e-4, 50 эпох, max_ar=4, static_channels=[7,8]

Обучение (6 эпох, потом остановили — хуже чем эксп. 1):
```
Epoch 4 (AR=1) with attention threshold 0.0
[Epoch 5] train_loss=0.11451  val_loss=0.26562  val_ACC=0.7055
Val loss reduced by 0.01041. Saving best model... 

Epoch 5 (AR=1) with attention threshold 0.0
[Epoch 6] train_loss=0.10896  val_loss=0.26389  val_ACC=0.7083
Val loss reduced by 0.00173. Saving best model... 
```

Вывод: 1029 mesh при hidden=128 — слишком мало ёмкости, сигнал размазывается.

---

## Эксперимент 3: 1029 mesh-узлов + большая модель

Конфиг: hidden=256, lr=1e-4, 80 эпох, max_ar=4, static_channels=[7,8], mesh_levels=[3,6], 16 MP steps

Изменения по сравнению с эксп. 1:
- hidden 128 → 256
- message_passing_steps 8 → 16
- mesh_levels [3,5] → [3,6] (259 → 1029 узлов)
- lr 3e-4 → 1e-4
- epochs 50 → 80, patience 10 → 15
- decoder hidden 64 → 128
- static_channels=[7,8] (z_surf, lsm не предсказываем)

Обучение (остановлен на ep14, AR ещё =1):
```
Epoch 13 (AR=1) with attention threshold 0.043392
[Epoch 14] train_loss=0.07343  val_loss=0.21289  val_ACC=0.7888
Val loss reduced by 0.00159. Saving best model... 
```

Инференс (predict.py --ar-steps 4 --max-samples 200):
```
=== Inference summary (200 samples) ===
Grid: 61x41 (G=2501) | C=19 | AR=4 horizons
Overall: MSE=0.297793 | RMSE=0.545704 | MAE=0.332887
Baseline: RMSE=0.489345 | MAE=0.284239
Skill: -11.52%
ACC: 0.6765 | base: 0.6883 | Δ=-0.0118

Per-horizon:
  +06h: RMSE=0.404344 | base=0.313348 | skill=-29.04% | ACC=0.8081 (base 0.8385)
  +12h: RMSE=0.480441 | base=0.454923 | skill=-5.61% | ACC=0.7166 (base 0.7072)
  +18h: RMSE=0.580301 | base=0.542019 | skill=-7.06% | ACC=0.6261 (base 0.6264)
  +24h: RMSE=0.678309 | base=0.599090 | skill=-13.22% | ACC=0.5552 (base 0.5810)

Per-horizon per-channel RMSE (physical units):
         var   unit      +6h     +12h     +18h     +24h
         t2m      K    2.78°C    3.82°C    4.90°C    5.97°C
         10u    m/s     0.63     0.92     1.30     1.64
         10v    m/s     0.58     0.91     1.27     1.55
         msl     Pa     1.59     3.10     4.84     6.40
       t@850      K    1.34°C    1.97°C    2.66°C    3.35°C
       u@850    m/s     1.59     2.42     3.39     4.31
       v@850    m/s     1.71     2.86     4.15     5.17
       z@850  m²/s²      0.9m      2.0m      3.2m      4.3m
       z@500  m²/s²      1.8m      3.8m      6.1m      8.2m
```

Вывод: Только 14 эпох (AR=1), модель не дошла до AR=2. Over-smoothing из-за плотного mesh (1029) + 16 MP steps.
Skill отрицательный, но модель на AR=1 была val_ACC=0.79. При AR rollout на 4 шага — деградация.
1029 mesh + 16 MP → over-smoothing. Mesh не должен быть слишком плотным для региона.

---

## Эксперимент 4: 259 mesh + hidden 256 + 6 MP + boundary mask

Конфиг: hidden=256, lr=2e-4, 60 эпох, max_ar=4, static_channels=[7,8], mesh_levels=[3,5], 6 MP steps, boundary_mask_width=5

Изменения по сравнению с эксп. 1:
- hidden 128 → 256, MP steps 8 → 6, lr 3e-4 → 2e-4
- static_channels=[7,8], boundary_mask_width=5 (inner 51×31)

Обучение (38 эпох, AR=3):
```
Epoch 38 (AR=3) val_loss=0.11801  val_ACC=0.7425
```

Инференс (200 samples, AR=4):
```
Overall: RMSE=0.510303 | Skill: -4.28%  (по всей сетке)
Inner zone (1581 pts): RMSE=0.365978 | Skill: 25.37%

Per-horizon (inner zone):
  +06h: skill=33.37% | t2m=2.95°C | ACC=0.8817
  +12h: skill=33.89% | t2m=3.43°C | ACC=0.8065
  +18h: skill=26.13% | t2m=4.13°C | ACC=0.7174
  +24h: skill=18.36% | t2m=4.69°C | ACC=0.6424
```

**Вывод: хуже эксп. 1.** Boundary mask навредил — модель не учится на 37% точек (краевая зона), а 6 MP вместо 8 ухудшил распространение сигнала. hidden=256 не скомпенсировал.

---

## Эксперимент 5: 23 канала + time forcing (hidden=256, 8 MP, без boundary mask)

Конфиг: hidden=256, lr=3e-4, 60 эпох, max_ar=4, 23 каналов
static_channels=[7,8], forcing_channels=[19,20,21,22], boundary_mask_width=0
mesh=[3,5] (259 узлов), 8 MP steps

Обучение (epoch 15, AR=1 → переход на AR=2):
```
val_loss=0.43529  val_ACC=0.5854  ← ОЧЕНЬ ПЛОХО (для сравнения: эксп.1 val_ACC=0.87)
```

Инференс (200 samples, AR=4):
```
Overall: Skill=33.78% (завышен из-за time channels в baseline)

Per-horizon per-channel RMSE (physical):
  t2m:   3.61°C → 3.95 → 4.70 → 5.62  (эксп.1: 2.66 → 3.14 → 3.69 → 4.33)
  u@850: 2.35   → 4.03 → 5.68 → 7.25  (эксп.1: 1.59 → 2.42 → 3.39 → 4.31)
  msl:   5.50   → 6.18 → 7.29 → 8.57  (эксп.1: 1.50 → 2.61 → 4.24 → 5.83)
```

**Вывод: значительно хуже эксп. 1 по всем метеоканалам.**

### Root Cause (найден после эксп. 5)

**Критический баг в `test()` (train.py):** функция валидации НЕ применяла carry-forward для static/forcing каналов.

В `train_epoch()` AR-цикл правильно делал carry-forward:
- static: `curr_state[:,:,ch] = curr_state[:,:,ch]` (копия из предыдущего шага)
- forcing: `curr_state[:,:,ch] = y_steps[step][:,:,ch]` (из таргета)

Но `test()` этого не делала! Результат:
- `outs[:,:,7] = X_last[:,:,7] + delta_pred[:,:,7]` — delta необучена (лосс=0) → мусор
- val_ACC считалась по ВСЕМ каналам включая мусорные → **занижена**
- Curriculum (переход AR=1→2→3→4) опирается на val_ACC → **запоздалый или неправильный переход**
- Early stopping тоже нарушен

Это объясняет почему ВСЕ эксперименты с `static_channels` хуже эксп. 1:
- Эксп. 1 (без static): test() корректна, val_ACC=0.87 — честная
- Эксп. 2-5 (с static): test() даёт занижённую val_ACC → curriculum сломан

**Фикс:** добавлен carry-forward в `test()` для static_channels (из X_last) и forcing_channels (из y_step0).

---

## Эксперимент 6: 23f + time forcing + static fix (PLAN)

Конфиг: архитектура эксп. 1 (hidden=128) + 23f данные + forcing + **исправленная test()**

- hidden=128, 8 MP steps, mesh [3,5] (259 узлов)
- lr=3e-4, 50 эпох, patience=10
- static_channels=[7,8], forcing_channels=[19,20,21,22], boundary_mask_width=0
- 23f dataset (19 метео + 4 time features), obs=2, pred=1
- decoder output_dim=23

Гипотеза: эксп. 5 провалился из-за бага в test() (val_ACC занижена → curriculum сломан), а NOT из-за time features.
С фиксом test() forcing channels должны работать корректно, time features дадут модели информацию о суточном/годовом цикле.
hidden=128 (а не 256 как в эксп. 5) — проверено, достаточно для региона.

Обучение (начало):
```
[Train] static_channels=[7, 8], forcing_channels=[19, 20, 21, 22]
[Train] → маска каналов: 17/23 dynamic (no-loss: [7, 8, 19, 20, 21, 22])
[Init] val_loss=0.19513 val_acc=83.1985   ← УЖЕ 83.2%! (эксп. 5 с багом: 58.5%)

Epoch 0 (AR=1): train_loss=0.14514  val_loss=0.12834  val_ACC=83.2333
Epoch 1 (AR=1): train_loss=0.11615  val_loss=0.10395  val_ACC=83.2555
```
val_ACC=83.2% с самого начала подтверждает что фикс test() работает корректно.