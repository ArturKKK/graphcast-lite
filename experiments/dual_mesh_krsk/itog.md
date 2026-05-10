# DualMesh Красноярск — итоги

## Конфигурация
- Глобальная модель: InteractionNet v2, 5.9M params (frozen)
- Региональный модуль: 938,515 params (hidden=256, 4 MP steps, shared weights)
- Сетка: 133,279 глобальных + 2,501 ROI (lat 50–60, lon 83–98)
- Региональный меш: level 8, 2,329 узлов (буфер 2°)
- Cross-edges: 13,974 (k=3, bidirectional)
- Encoding: 9,316 рёбер (mesh-centric KNN, k=4)
- Decoding: 7,503 рёбер (grid-centric KNN, k=3)
- lr=5e-4, 30 эпох, batch_size=1, ROI-only loss
- use_residual=false, zero-init decoder output layer

## Результаты инференса (200 тест сэмплов, +6h)

| Метрика    | DualMesh | freeze6 (baseline) |
|------------|----------|---------------------|
| Skill      | 71.31%   | 75.82%              |
| ACC        | 0.9691   | ~0.97               |
| t2m RMSE   | 1.16°C   | 0.96°C              |
| msl RMSE   | 0.67 Pa  | 0.63 Pa             |
| 10u RMSE   | 0.41 m/s | 0.43 m/s            |
| 10v RMSE   | 0.37 m/s | 0.35 m/s            |

## Обучение (лог)
```
epoch  train_loss    val_loss   val_ACC
    1    0.007044    0.007520    0.9657
    7    0.006822    0.007459    0.9657
```
Loss падает медленно, ACC за 30 эпох не двигается.

## Вывод
DualMesh **хуже** freeze6 на 4.5% skill и 0.2°C по t2m.
Региональный модуль фактически не вносит полезную коррекцию — 
global_pred проходит "as is", а correction ≈ 0.

## Баги исправленные в процессе (4 ретрейна)
1. Loss по всем 133K точкам → gradient dilution ~48x → ROI-only loss
2. In-place combine `output[mask] += corr` → functional zeros_like + add
3. 40% mesh-узлов без encoding рёбер (dead nodes) → mesh-centric KNN
4. Random init decoder output → noise вместо нулевой коррекции → zero-init
