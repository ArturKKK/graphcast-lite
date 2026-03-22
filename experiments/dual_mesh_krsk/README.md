# Dual Mesh: Запуск на кластере

## 1. Обучение

```bash
# SSH на кластер
ssh user@cluster

# Переход в проект
cd graphcast-lite

# Активация окружения (если conda)
conda activate graphcast
# или: source .venv/bin/activate

# Обучение DualMeshModel
# ROI: Красноярский край (lat 50-60, lon 83-98)
nohup python scripts/train_dual_mesh.py experiments/dual_mesh_krsk \
  --pretrained experiments/multires_nores_freeze6/results/best_model.pth \
  --data-dir data/datasets/multires_krsk_19f \
  --roi 50 60 83 98 \
  --epochs 30 \
  --lr 5e-4 \
  --reg-mesh-level 7 \
  --reg-steps 4 \
  --cross-k 3 \
  --hidden-dim 256 \
  > experiments/dual_mesh_krsk/log.log 2>&1 &

# Мониторинг
tail -f experiments/dual_mesh_krsk/log.log
# или:
tail -f experiments/dual_mesh_krsk/results/training_log.txt
```

## 2. Инференс

```bash
python scripts/predict_dual_mesh.py experiments/dual_mesh_krsk \
  --pretrained experiments/multires_nores_freeze6/results/best_model.pth \
  --regional-ckpt experiments/dual_mesh_krsk/results/best_regional.pth \
  --data-dir data/datasets/multires_krsk_19f \
  --roi 50 60 83 98 \
  --max-samples 200
```

## 3. Параметры для экспериментов

| Параметр | По умолчанию | Варианты |
|---|---|---|
| `--reg-mesh-level` | 7 (~467 вершин) | 8 (~1800 вершин, медленнее) |
| `--reg-steps` | 4 | 6, 8 |
| `--cross-k` | 3 | 5 |
| `--hidden-dim` | 256 | 128 (быстрее), 512 (больше параметров) |
| `--lr` | 5e-4 | 1e-4, 1e-3 |
| `--reg-buffer` | 2.0° | 3.0°, 5.0° |

## 4. Архитектура

```
Global Grid (133K) ──► Global Encoder ──► Global Mesh (40K, frozen)
                                              ↕ cross-edges (3 NN, bidirectional)
ROI Grid (2.5K)    ──► Regional Encoder ──► Regional Mesh (467, level 7)
                                              ↓ 4 InteractionNet steps
                                         Regional Decoder ──► ROI correction
                                              ↓
Final = global_pred + regional_correction (in ROI)
```

- **Региональные параметры**: ~1.2M (vs 6M глобальных)
- **Глобальная модель**: полностью заморожена
- **Вход регионального энкодера**: raw features + global grid latents (из encoder глобальной модели)

## 5. Файлы

| Файл | Описание |
|---|---|
| `src/dual_mesh.py` | DualMeshModel, региональный меш, cross-edges |
| `scripts/train_dual_mesh.py` | Скрипт обучения |
| `scripts/predict_dual_mesh.py` | Скрипт инференса |
| `experiments/dual_mesh_krsk/config.json` | Конфиг (глобальной модели) |
| `experiments/dual_mesh_krsk/results/best_regional.pth` | Региональные веса |
| `experiments/dual_mesh_krsk/results/best_model.pth` | Полные веса |
