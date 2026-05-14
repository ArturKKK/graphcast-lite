# Russia 33f multires — finetune от v3 GLOBAL, без roi_only_loss

**Гипотеза:** v3 GLOBAL (33ch) уже выучила сильную физическую базу. Если на её основе дообучить мультирезолюционную модель Russia (та же 33-канальная архитектура + плотная сетка 0.25° в ROI), но БЕЗ `roi_only_loss`, мы получим:
- (а) когерентный глобальный контекст (loss по всем узлам — модель не дрейфит вне ROI);
- (б) повышенное разрешение в регионе через mesh;
- (в) перенос всех улучшений v3 (plev@250/1000, time-forcing, noise, channel weights).

**Архитектурно идентично `wb2_512x256_33f_ar_v3`**, отличия только:
- `data_dir → /data/datasets/multires_russia_33f` (новый, см. `scripts/build_multires_russia_33f.py`)
- `num_epochs: 80 → 40` (finetune, не from-scratch)
- `learning_rate: 3e-4 → 1e-4`
- `freeze_processor_epochs: 0 → 6`, `lr_factor: 1.0 → 0.1`
- `max_ar_steps: 8 → 4` (упор на короткий горизонт, как у Russia freeze6)
- `pred_window_used: 8 → 1`
- `roi_only_loss=false` (по умолчанию, не выставлен)

**Pretrained:** `experiments/wb2_512x256_33f_ar_v3/best_model.pth` (val 0.01865, AR=3, ep 30).

**Запуск:** см. `scripts/_mlc_run_russia_33f_v4.sh`.

## Что должно случиться

| Сценарий | Russia 19f roi-only (текущий) | Russia 19f no-roi | Russia 33f no-roi (этот) |
|---|---|---|---|
| Loss domain | только ROI | весь grid | весь grid |
| AR-rollout стабильность | низкая (drift вне ROI заражает ROI) | средняя | **высокая** (33ch + v3 база) |
| Метрика на ROI | val 0.01292 (AR=3) | ? | ? |
| Метрика глобально | мусор (RMSE 0.31) | разумная | **должна быть лучшая** |
| Финетюн от | global v2 19f | global v2 19f | **global v3 33f** |

Ожидается, что 33f-вариант даст лучшую метрику и на регионе, и глобально, потому что:
1. Стартует с уже более сильной базы (v3 > v2);
2. Имеет +10 plev-каналов для лучшей вертикальной структуры;
3. Не страдает roi-only AR-feedback дрейфом;
4. Плотная сетка 0.25° даёт фактическое преимущество над v3 GLOBAL на регионе.
