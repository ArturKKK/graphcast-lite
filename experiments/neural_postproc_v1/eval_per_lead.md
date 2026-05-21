# Neural postproc v1 — per-lead evaluation

Trained 2026-05-21 on v3. Corpus: 50 stations, init dates 2018-2020 every 5 days, lead 6/12/18/24h, 33-ch GNN AR rollout.
Train=284 692 (2018-2019), val=142 800 (2020). MLP 28→128→128, residual on top of GNN.
Best checkpoint: epoch 27 (early-best by val combined).

## All leads (val 2020, n=142 800)

|  | T2m RMSE °C | T2m bias °C | T2m MAE °C | Wind vec-RMSE m/s | Wind speed-RMSE m/s |
|---|---:|---:|---:|---:|---:|
| **postproc v1** | **1.810** | −0.048 | 1.331 | **2.067** | **1.490** |
| GNN raw         |   2.105   | −0.436 | 1.558 |   2.557   |   1.833   |
| Δ (improvement) | **−0.295**| +0.388 | −0.227| **−0.490**| **−0.343**|

## Per lead_h

| lead | n | T2m RMSE pp | T2m RMSE GNN | Δ T2m | T2m bias pp | T2m bias GNN | Wind vec pp | Wind vec GNN | Δ wind | Wind speed pp | Wind speed GNN |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  6 | 35 701 | 1.687 | 1.958 | −0.271 | −0.003 | −0.326 | 1.975 | 2.469 | −0.495 | 1.431 | 1.796 |
| 12 | 35 746 | 1.800 | 2.089 | −0.290 | −0.033 | −0.387 | 2.074 | 2.564 | −0.491 | 1.496 | 1.827 |
| 18 | 35 648 | 1.839 | 2.136 | −0.298 | −0.070 | −0.487 | 2.065 | 2.566 | −0.500 | 1.487 | 1.844 |
| 24 | 35 705 | 1.907 | 2.229 | −0.321 | −0.087 | −0.544 | 2.150 | 2.627 | −0.477 | 1.544 | 1.863 |

## Notes / known issues
- **Corpus только до lead=24h** (3 шага AR). v2 расширим до 120h (20 шагов).
- Static фичи `lsm=1.0`, `dist_to_coast=0`, `urban_flag=0`, `z_surf=elev` — мёртвые/дубликаты. Будут починены из ERA5 static fields в v2.
- `dewpoint_depression`: min=22.5°C — формула битая (использует q850 как surface dewpoint). v2: считать через RH из t2m+q1000.
- Биас GNN растёт от −0.33 (6h) до −0.54°C (24h) — постпроц гасит до ≤0.1°C на всех лидах.
