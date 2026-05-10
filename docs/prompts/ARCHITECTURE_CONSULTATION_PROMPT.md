# Consultation: Improving Regional Resolution in a GraphCast-style GNN Weather Model

## Context

I'm building **GraphCast-lite** — a simplified version of Google DeepMind's GraphCast for weather prediction. The model is working and producing good results. I need architectural advice on improving regional (local) forecast quality.

## Current Architecture

### Overview

The model follows the **Encode → Process → Decode** pattern on an irregular graph:

```
Grid nodes (weather data)
    ↓  Encoder (MLP + GCNConv, bipartite grid→mesh edges)
Mesh nodes (icosahedral sphere, uniform)
    ↓  Processor (12 InteractionNet steps, mesh↔mesh edges)
Mesh nodes (updated)
    ↓  Decoder (MLP + GCNConv, bipartite mesh→grid edges)
Grid nodes (predicted weather at next timestep)
```

### Grid (Input/Output)

- Regular lat/lon grid: **512×256 = 131,072 nodes**, ~0.7° resolution
- Each node has **19 weather variables**: t2m, 10u, 10v, msl, tp, sp, tcwv, z_surf, lsm, t/u/v/z/q @850hPa, t/u/v/z/q @500hPa
- Input: 2 observation timesteps (t-6h, t) → 38 dynamic features + 5 static (lat, lon, x, y, z)
- Output: 4 prediction steps (+6h, +12h, +18h, +24h) → 76 channels = 19 vars × 4 steps
- Training: MSE loss, autoregressive curriculum (1→4 steps over epochs)

### Mesh (Latent Space)

- **Hierarchical icosahedron** with levels [4, 6]:
  - Level 4: 2,562 vertices (~4.4° spacing) — coarse long-range edges
  - Level 6: 40,962 vertices (~1.1° spacing) — fine local edges
  - Both levels' edges are **merged into one graph** (multi-scale connectivity)
- Node count formula: V(L) = 10·4^L + 2

### Edge Features

Every edge (mesh↔mesh) carries **4 features**:
1. Normalized chordal distance (L2 in 3D, divided by global max edge distance)
2-4. Normalized relative position (dx, dy, dz) in the **receiver's local coordinate system**

These are projected to latent dimension by a linear layer, then updated at each InteractionNet step with residual connections.

### Encoder

```
MLP: Linear(43→256) → PReLU → Linear(256→256) → PReLU → Linear(256→256) → LayerNorm
GCN: GCNConv(256→256) → SiLU → GCNConv(256→256) → SiLU
```

- Operates on bipartite **grid→mesh** edges
- Grid→mesh edges: every grid node connects to all mesh nodes within radius = 0.6 × max_mesh_edge_distance (KDTree query)
- After encoding, only mesh node representations are passed to the processor

### Processor (InteractionNet, 12 steps with UNSHARED weights)

Each `InteractionNetLayer`:
```
Edge update:  e'_ij = MLP([h_i || h_j || e_ij]) + e_ij     (residual)
Aggregation:  ē_j  = mean({e'_ij : i ∈ N(j)})              (scatter_mean)
Node update:  h'_j  = MLP([h_j || ē_j]) + h_j              (residual)
LayerNorm on both nodes and edges after each step
```

MLPs are 2-layer: Linear(in→256) → Swish → Linear(256→out)

Edge features are projected once from 4D → 256D, then updated jointly with node features through all 12 steps.

### Decoder

Same MLP+GCN structure as encoder, but on **mesh→grid** edges.
Mesh→grid edges: each grid node connects to the 3 vertices of the mesh triangle it falls inside (barycentric).

### Model Stats
- ~5.9M parameters total
- Encoder: ~341K, Decoder: ~150K, Processor: ~5.4M
- Activation: Swish (SiLU)
- Training: lr=3e-4, 80 epochs, Adam

---

## The Regional Problem

I want better forecasts for a specific region: **Krasnoyarsk, Russia** (55.5-56.5°N, 92-94°E).

### What I've Done: Multi-resolution Grid + Fine-tuning

1. Created a **multi-resolution grid dataset**:
   - Kept the global 0.7° grid everywhere EXCEPT inside the region of interest (ROI)
   - Replaced ROI points with **0.25° grid** (regional ERA5 data)
   - Result: **133,279 grid nodes** (131K global outside ROI + 2,501 regional inside ROI)
   - ~7.6× densification in the ROI

2. **Fine-tuned** the pretrained global model on this multi-res dataset:
   - Froze the processor (12 InteractionNet layers) for 6 epochs
   - Then unfroze with LR×0.1 for the processor
   - Total: 32 epochs, lr=1e-4
   - No residual connection (model predicts full field, not increment)

### Results

| Metric | Global model (before finetune) | After multires finetune |
|---|---|---|
| Global Skill | 57.26% | 66.94% |
| Regional Skill (45 nodes) | ~40% | **75.82%** |
| Regional t2m RMSE @+6h | ~1.8°C | **0.96°C** |
| Regional t2m RMSE @+24h | ~3.5°C | **1.40°C** |

Good results! But I believe there's a **fundamental bottleneck**.

### The Bottleneck

The mesh is NOT modified — it's the same global icosphere (40,962 nodes). Over the Krasnoyarsk ROI (~1°×2°), there are only **~20-30 mesh nodes** with ~1.1° spacing.

So the information flow is:
```
2,501 dense grid points (0.25°)
    ↓  Encoder (scatter/aggregate)
~25 mesh nodes (1.1° spacing)     ← BOTTLENECK: 100:1 compression!
    ↓  Processor (12 steps)
~25 mesh nodes
    ↓  Decoder (interpolate)
2,501 dense grid points
```

The encoder aggregates 2,501 points into ~25 mesh nodes, losing fine-grained spatial information. The processor works with the same ~25 mesh nodes regardless of whether the grid is 0.7° or 0.25°. The decoder must reconstruct 0.25° detail from ~25 mesh nodes.

---

## Proposed Solutions — Need Your Advice

### Option A: Densify the mesh in ROI

Add finer icosphere levels (7 or 8) **only inside the ROI**:

- Level 7 in ROI: ~600 additional mesh vertices at ~0.55° spacing
- Level 8 in ROI: ~2,400 additional mesh vertices at ~0.28° spacing

This would make the mesh density comparable to the 0.25° grid in the ROI.

**My concerns:**
- The icosahedron's virtue is uniform spacing. Adding level-8 vertices only in ROI creates heterogeneous edge lengths. However, the model already handles multi-scale edges (levels 4+6 have edges differing 4×), and edge features encode distance explicitly.
- Need to retrain/finetune the processor since graph topology changes.

### Option B: Skip-connection around the mesh (local grid-to-grid GNN)

Add a **local processor** operating directly on the dense grid nodes in the ROI:

```
Grid → Mesh → Processor → Mesh → Grid
  |                                 ↑
  └──── Local GNN (2-3 layers) ─────┘
        (only on 2,501 ROI nodes, 
         0.25° neighborhood edges)
```

- Simple to implement (add after decoder)
- Doesn't modify the main architecture
- But only captures local interactions, no long-range context improvement

### Option C: Something else?

- Attention-based encoder that can better aggregate dense grid→sparse mesh?
- Hierarchical encoder with separate regional/global passes?
- Completely different approach?

---

## Questions

1. **Is Option A (mesh densification in ROI) architecturally sound?** Given that the InteractionNet already handles multi-scale edges, will adding much finer edges in one region cause problems (training instability, gradient issues, etc.)?

2. **Is this bottleneck real and significant?** Or can the 12-step InteractionNet processor theoretically reconstruct fine-grained spatial patterns from ~25 mesh nodes through its message passing?

3. **What's the best approach** to improve regional resolution without sacrificing global performance? Are there approaches from the literature I should consider?

4. **Any other architectural ideas** for handling multi-resolution grids in GNN-based weather models?

## Constraints

- I can retrain/finetune the model (have access to GPU cluster)
- Total parameter budget: up to ~10M (currently 5.9M)
- Cannot change the input data format (ERA5 reanalysis, 19 variables)
- Must keep autoregressive rollout capability (1→4 steps)
- PyTorch + PyTorch Geometric stack
