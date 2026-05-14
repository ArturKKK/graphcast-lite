# Russia 19f multires — БЕЗ roi_only_loss

**Гипотеза:** roi_only_loss=true даёт хорошую метрику на регионе, но теряет физическую согласованность вне ROI — а это вредит AR-rollout (вне ROI дрейф → через graph message passing засоряет ROI → ошибки растут экспоненциально).

**Что меняется vs `multires_russia_19f_freeze6`:**
- `roi_only_loss: true → false` (loss теперь по всем 264948 узлам)
- `num_epochs: 32 → 40` (больше запас на конвергенцию полного-grid loss)
- Всё остальное идентично freeze6

**Pretrained:** finetune от глобальной v2 19f (experiments/wb2_512x256_19f_ar_v2/best_model.pth), freeze processor 6 эпох, lr_factor=0.1.

**Сравнение, которое ожидается после обучения:** какой подход даёт лучше RMSE/ACC на ROI:
- (A) roi_only_loss=true (текущий freeze6, val 0.01292 на ROI@AR=3)
- (B) roi_only_loss=false (этот эксперимент)

**Запуск на VM (graphcast_v3-z1w6to):** см. `scripts/_mlc_run_russia_noroi.sh`.
