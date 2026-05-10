# Experiment Results — Diploma Benchmarks
## Date: 2026-04-13

All results collected during benchmark runs on MLC VMs.

---

## 1. Freeze6 Control (GNN-only, multires 133K grid)
**Status**: DONE
**Config**: multires_nores_freeze6, --no-residual, --prune-mesh, --ar-steps 4, --per-channel, --max-samples 200
**VM**: graphcast_v2-bm2c9x, A100 80GB
**Predictions**: /data/predictions/predictions_control_freeze6.pt (200×133279×76)

### Global (133279 nodes)
| Metric | Value |
|--------|-------|
| Skill | 66.94% |
| RMSE | 0.179174 |
| ACC | 0.9832 |
| MAE | 0.077980 |

Per-horizon global:
| Horizon | RMSE | Skill | ACC |
|---------|------|-------|-----|
| +06h | 0.120926 | 66.97% | 0.9925 |
| +12h | 0.156564 | 69.43% | 0.9874 |
| +18h | 0.197443 | 66.90% | 0.9795 |
| +24h | 0.224264 | 65.50% | 0.9733 |

### Region (2501 ROI nodes, Krasnoyarsk)
| Metric | Value |
|--------|-------|
| Skill | 75.16% |
| RMSE | 0.100301 |
| ACC | 0.9492 |

Per-horizon region:
| Horizon | RMSE | Skill | ACC |
|---------|------|-------|-----|
| +06h | 0.074739 | 71.45% | 0.9690 |
| +12h | 0.091580 | 76.02% | 0.9557 |
| +18h | 0.107449 | 76.03% | 0.9416 |
| +24h | 0.121340 | 75.06% | 0.9304 |

Region t2m RMSE (physical):
| +6h | +12h | +18h | +24h |
|------|------|------|------|
| 1.15°C | 1.47°C | 1.66°C | 1.80°C |

## 2. Cascade GNN→UNet
**Status**: DONE
**Config**: predict_cascade.py, freeze6 → downscaler_v2_krsk, --ar-steps 4, --max-samples 200
**Note**: Cascade does NOT improve over GNN-only on 0.25° ERA5 ground truth!

### Key finding: cascade Δ vs GNN-only (mean over all vars)
| Horizon | Mean Δ |
|---------|--------|
| +06h | -0.8% |
| +12h | -0.1% |
| +18h | -0.0% |
| +24h | -0.1% |

### t2m cascade vs GNN (physical RMSE):
| Horizon | GNN | Cascade | Δ |
|---------|-----|---------|---|
| +6h | 1.67°C | 1.69°C | -1.2% |
| +12h | 1.95°C | 1.97°C | -1.4% |
| +18h | 2.11°C | 2.14°C | -1.2% |
| +24h | 2.24°C | 2.27°C | -1.4% |

### Skill vs persistence (0.25° ERA5):
| Horizon | Mean Skill |
|---------|------------|
| +06h | +20.4% |
| +12h | +40.3% |
| +18h | +45.0% |
| +24h | +45.0% |

### Variables where cascade helps (>0.5%):
- msl: +1.4% to +4.6%
- z@850: +2.1% to +3.4%
- t@850: +0.6% to +0.8% (at +12h-24h)

### Variables where cascade hurts (<-1%):
- t2m: -1.2% to -1.4%
- tcwv: -0.9% to -2.8%
- t@500: -2.1% to -2.4%
- z@500: -7.6% (at +6h)

## 3. Timeforce 23f (regional 61×41)
**Status**: DONE
**Config**: region_krsk_23f_timeforce, --no-residual, --prune-mesh, --ar-steps 4, --per-channel, --max-samples 200
**Note**: Very weak model, time features did not help much

### Global (2501 nodes, 61×41 regional grid)
| Metric | Value |
|--------|-------|
| Skill | 17.88% |
| RMSE | 0.412878 |
| ACC | 0.6162 |

Per-horizon:
| Horizon | RMSE | Skill | ACC |
|---------|------|-------|-----|
| +06h | 0.274017 | 17.30% | 0.7549 |
| +12h | 0.383304 | 19.04% | 0.6480 |
| +18h | 0.454416 | 18.26% | 0.5603 |
| +24h | 0.503360 | 17.05% | 0.5015 |

### Inner region (1581 nodes, bmw=5):
| Metric | Value |
|--------|-------|
| Skill | 18.17% |
| ACC | 0.6034 |

t2m RMSE (physical, inner region):
| +6h | +12h | +18h | +24h |
|------|------|------|------|
| 3.94°C | 4.09°C | 4.78°C | 5.46°C |

---

## 4. Data Assimilation (DA) Experiments
**Base model**: multires_nores_freeze6 (same as control)
**Setup**: --obs-sparsity 0.1 --obs-roi-only (250 inline stations from 2501 ROI nodes, seed=42)
**All runs**: --ar-steps 4 --per-channel --max-samples 200

### 4.1 Summary Table — Region (2501 ROI nodes)

| Method | Parameters | Region Skill | Δ Skill | t2m +6h | t2m +12h | t2m +18h | t2m +24h |
|--------|-----------|-------------|---------|---------|----------|----------|----------|
| Control (no DA) | — | 75.16% | — | 1.15°C | 1.47°C | 1.66°C | 1.80°C |
| Nudging offline | α=0.3 s=0.1 | 75.79% | +0.63pp | 1.12°C | 1.44°C | 1.62°C | 1.76°C |
| Nudging sequential | α=0.3 s=0.1 | 75.95% | +0.79pp | 1.12°C | 1.43°C | 1.61°C | 1.74°C |
| Nudging sequential | α=0.5 s=0.1 | 76.29% | +1.13pp | 1.11°C | 1.41°C | 1.58°C | 1.71°C |
| OI | corr=5km σ_o=0.3 s=0.1 | 76.65% | +1.49pp | 1.09°C | 1.40°C | 1.56°C | 1.69°C |
| **OI** | **corr=10km σ_o=0.5 s=0.1** | **76.85%** | **+1.69pp** | **1.09°C** | **1.38°C** | **1.55°C** | **1.68°C** |

### 4.2 Nudging α=0.3 offline (s=0.1)
File: /data/predictions/nudge_s01_a03_offline_ar4.txt

Global: Skill=66.94%, RMSE=0.179148, ACC=0.9832
Region: Skill=75.79%, RMSE=0.097741

Region per-horizon:
| Horizon | Skill | t2m |
|---------|-------|-----|
| +06h | 72.18% | 1.12°C |
| +12h | 76.63% | 1.44°C |
| +18h | 76.64% | 1.62°C |
| +24h | 75.70% | 1.76°C |

### 4.3 Nudging α=0.3 sequential (s=0.1)
File: /data/predictions/nudge_s01_a03_seq_ar4.txt

Global: Skill=66.94%, RMSE=0.179140
Region: Skill=75.95%, RMSE=0.097113

Region per-horizon:
| Horizon | Skill | t2m |
|---------|-------|-----|
| +06h | 72.18% | 1.12°C |
| +12h | 76.73% | 1.43°C |
| +18h | 76.82% | 1.61°C |
| +24h | 75.92% | 1.74°C |

### 4.4 Nudging α=0.5 sequential (s=0.1)
File: /data/predictions/nudge_s01_a05_seq_ar4.txt

Global: Skill=66.94%, RMSE=0.179125
Region: Skill=76.29%, RMSE=0.095711

Region per-horizon:
| Horizon | Skill | t2m |
|---------|-------|-----|
| +06h | 72.53% | 1.11°C |
| +12h | 77.05% | 1.41°C |
| +18h | 77.16% | 1.58°C |
| +24h | 76.29% | 1.71°C |

### 4.5 OI (corr_len=10000, σ_o=0.5, s=0.1)
File: /data/predictions/oi_s01_corr10k_sig05_ar4.txt

Global: Skill=66.95%, RMSE=0.179101
Region: Skill=76.85%, RMSE=0.093463

Region per-horizon:
| Horizon | Skill | t2m |
|---------|-------|-----|
| +06h | 73.07% | 1.09°C |
| +12h | 77.55% | 1.38°C |
| +18h | 77.71% | 1.55°C |
| +24h | 76.89% | 1.68°C |

### 4.6 OI (corr_len=5000, σ_o=0.3, s=0.1)
File: /data/oi_corr5000.log

Global: Skill=66.98%, RMSE=0.179109
Region: Skill=76.65%, RMSE=0.094253

Region per-horizon:
| Horizon | Skill | t2m |
|---------|-------|-----|
| +06h | 72.87% | 1.09°C |
| +12h | 77.37% | 1.40°C |
| +18h | 77.52% | 1.56°C |
| +24h | 76.68% | 1.69°C |

### 4.7 Key Findings
1. **OI > Nudging**: OI дает лучшую коррекцию (+1.69pp vs +1.13pp для nudging α=0.5)
2. **Sequential > Offline**: Sequential nudging стабильно лучше offline (при одинаковом α)
3. **Больший α → лучше**: α=0.5 > α=0.3 для nudging sequential
4. **DA улучшает регион, не трогает глобал**: Global Skill практически не меняется (66.94-66.95%), коррекция локализована в ROI
5. **t2m улучшение**: OI снижает t2m RMSE на 5-7% по всем горизонтам

## 4. Data Assimilation — Nudging
**Status**: pending
**Sparsity**: 1% and 10% stations
**Parameters to sweep**: alpha ∈ {0.01, 0.05, 0.1, 0.3}

## 5. Data Assimilation — OI
**Status**: pending
**Sparsity**: 1% and 10% stations
**Parameters to sweep**: sigma_obs, L_corr

## 6. DA + Cascade
**Status**: pending

---

## Previous Session Results (from memory, need re-verification):

### Freeze6 Control (200 samples):
- Global: Skill=66.94%, RMSE=0.179, ACC=0.9832
- Region (2501 nodes): Skill=75.16%, ACC=0.9492
- Region t2m: +6h=1.15°C, +12h=1.47°C, +18h=1.66°C, +24h=1.80°C

### Cascade GNN→UNet (200 samples):
- Mean improvement over GNN-only: +6h=12.4%, +12h=10.4%, +18h=9.3%, +24h=9.0%
- Cascade t2m: +6h=1.44°C, +24h=2.10°C
- Skill vs persistence: +6h=64.2%, +12h=70.1%, +18h=70.1%, +24h=69.1%

### Timeforce 23f (200 samples):
- Skill=17.88%, ACC=0.6162, t2m +6h=4.09°C, +24h=5.71°C (weak model)
