#!/usr/bin/env python3
"""Диагностика: проверяем порядок осей data vs grid в модели."""
import numpy as np

# Data хранится как (time, lon, lat, feat)
# Dataloader делает: X_frames.transpose(1,2,0,3) → (lon, lat, obs, feat)
#                     → reshape(lon*lat, obs*feat)
# => node index = lon_idx * n_lat + lat_idx   (LON-major)

# Модель строит позиции через meshgrid:
#   grid_lon, grid_lat = np.meshgrid(longs, lats)
#   → shape (n_lat, n_lon), flatten → node index = lat_idx * n_lon + lon_idx  (LAT-major)

# Это РАЗНЫЙ порядок! Node k в данных и node k в grid — РАЗНЫЕ точки.

n_lat, n_lon = 41, 61
print(f"grid: {n_lon}x{n_lat} = {n_lon*n_lat} nodes")

# Загрузим реальные координаты
coords = np.load('data/datasets/region_krsk_cds_19f/coords.npz')
lats = coords['latitude']   # (41,) — от 50 до 60
lons = coords['longitude']  # (61,) — от 83 до 98

print(f"lats: {lats[0]:.2f} → {lats[-1]:.2f} (n={len(lats)})")
print(f"lons: {lons[0]:.2f} → {lons[-1]:.2f} (n={len(lons)})")

# Порядок данных (lon-major): node k → lon_idx = k // n_lat, lat_idx = k % n_lat
# Порядок grid (lat-major): node k → lat_idx = k // n_lon, lon_idx = k % n_lon

print("\n=== Первые 5 узлов ===")
print(f"{'k':>5s}  {'data_lat':>10s} {'data_lon':>10s}  │  {'grid_lat':>10s} {'grid_lon':>10s}  │  {'match?':>6s}")
for k in range(5):
    # Данные (lon-major)
    d_lon_idx = k // n_lat
    d_lat_idx = k % n_lat
    data_lat = lats[d_lat_idx]
    data_lon = lons[d_lon_idx]
    
    # Grid (lat-major, meshgrid default)
    g_lat_idx = k // n_lon
    g_lon_idx = k % n_lon
    grid_lat = lats[g_lat_idx]
    grid_lon = lons[g_lon_idx]
    
    match = "OK" if (d_lat_idx == g_lat_idx and d_lon_idx == g_lon_idx) else "WRONG"
    print(f"{k:5d}  {data_lat:10.2f} {data_lon:10.2f}  │  {grid_lat:10.2f} {grid_lon:10.2f}  │  {match:>6s}")

# Покажем разницу более наглядно
n_wrong = 0
max_dist = 0.0
for k in range(n_lon * n_lat):
    d_lon_idx = k // n_lat
    d_lat_idx = k % n_lat
    g_lat_idx = k // n_lon
    g_lon_idx = k % n_lon
    if d_lat_idx != g_lat_idx or d_lon_idx != g_lon_idx:
        n_wrong += 1
        dlat = abs(lats[d_lat_idx] - lats[min(g_lat_idx, n_lat-1)])
        dlon = abs(lons[min(d_lon_idx, n_lon-1)] - lons[min(g_lon_idx, n_lon-1)])
        max_dist = max(max_dist, (dlat**2 + dlon**2)**0.5)

print(f"\nMismatched nodes: {n_wrong}/{n_lon * n_lat} ({100*n_wrong/(n_lon*n_lat):.1f}%)")
print(f"Max position error: {max_dist:.2f}°")

# А теперь для тренировочного grid (512×256)
n_lat_g, n_lon_g = 256, 512
n_wrong_g = 0
for k in range(min(n_lon_g * n_lat_g, 100000)):
    d_lon_idx = k // n_lat_g
    d_lat_idx = k % n_lat_g
    g_lat_idx = k // n_lon_g
    g_lon_idx = k % n_lon_g
    if d_lat_idx != g_lat_idx or d_lon_idx != g_lon_idx:
        n_wrong_g += 1
n_tot = min(n_lon_g * n_lat_g, 100000)
print(f"\nТренировочный grid 512×256: {n_wrong_g}/{n_tot} mismatched ({100*n_wrong_g/n_tot:.1f}%)")
