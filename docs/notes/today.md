Created 2 todos

Let me verify the key detail — what data the training actually uses (interpolated vs real): 

Understand multires dataset building

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/scripts/build_multires_dataset.py#1-1), lines 1 to 300

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/multires_nores_freeze6)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/scripts/build_dataset_wb2.py)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/scripts/build_region_wb2.py)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/scripts/build_multires_dataset.py#300-300), lines 300 to 400

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/global_512x256_19f_2010-2021_07deg)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/region_krsk_61x41_19f_2010-2020_025deg)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/region_krsk_cds_19f_10y)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/multires_nores_freeze6/config.json#1-1), lines 1 to 50

Searched for regex `interpolat|merge|bilinear|multires_krsk`, 40 results

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/global_512x256_19f_jan2023_07deg)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/region_krsk_61x41_19f_jan2023_025deg)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/region_krsk_61x41_23f_2010-2020_025deg)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/temp.md#20-20), lines 20 to 50

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/multires_nores_freeze6/config.json#50-50), lines 50 to 72

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/global_512x256_19f_2010-2021_07deg/dataset_info.json)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/region_krsk_61x41_19f_2010-2020_025deg/dataset_info.json)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/multires_krsk_19f)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/multires_nores_nofreeze)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/global_512x256_19f_jan2023_07deg/dataset_info.json)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/region_krsk_61x41_19f_jan2023_025deg/dataset_info.json)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/data/datasets/region_krsk_61x41_23f_2010-2020_025deg/dataset_info.json)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/multires_krsk_19f/config.json)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/experiments/multires_nores_nofreeze/config.json)

Search complete: Find graph builder code

Searched for text `graph builder`, no results

Searched for regex `mesh2grid|grid2mesh`, 100 results

Searched for regex `multires.*flat|flat.*multires`, 5 results

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/src/create_graphs.py#1-1), lines 1 to 80

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/src/config.py#1-1), lines 1 to 120

Searched for files matching `**/src/**/*.py`, 29 matches

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/src/create_graphs.py#80-80), lines 80 to 296

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/src/mesh/grid_mesh_connectivity.py)

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/src/mesh/__init__.py)

Searched for regex `multires_krsk_19f_jan2023|jan2023.*merge|mode.*merge`, 11 results

Search complete: Find dataset loading for multires

Searched for text `multires`, 136 results

Searched for text `data.npy`, 64 results

Searched for text `obs_window`, 200 results

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/src/data/dataloader_chunked.py#1-1), lines 1 to 300

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/src/main.py#70-70), lines 70 to 130

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/src/config.py#50-50), lines 50 to 80

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/scripts/build_multires_dataset.py#1-1), lines 1 to 260

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/diploma_experiment_results.md#180-180), lines 180 to 210

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/PRESENTATION_CONTEXT.md#40-40), lines 40 to 60

Searched for regex `n_nodes|n_global_kept|n_regional` (`**/*.md`), no results

Ran terminal command:  cd /Users/a.s.tabakov/Developer/graphcast-lite && python3 -c "
import numpy as np

# Global grid
g_nlat, g_nlon = 256, 512
g_total = g_nlat * g_nlon
print(f'Global grid: {g_nlat} x {g_nlon} = {g_total} nodes')
print(f'Global resolution: ~0.703 deg (180/256)')

# ROI: lat 50..60, lon 83..98
# Global grid spacing
lat_spacing = 180.0 / 256  # ~0.703 deg
lon_spacing = 360.0 / 512  # ~0.703 deg
print(f'Global lat spacing: {lat_spacing:.4f} deg')
print(f'Global lon spacing: {lon_spacing:.4f} deg')

# Count global nodes in ROI
g_lats = np.linspace(-90, 90, 256)
g_lons = np.linspace(0, 360, 512, endpoint=False)
g_lat_in = (g_lats >= 50) & (g_lats <= 60)
g_lon_in = (g_lons >= 83) & (g_lons <= 98)
n_lat_roi = g_lat_in.sum()
n_lon_roi = g_lon_in.sum()
n_removed = n_lat_roi * n_lon_roi
print(f'Global nodes in ROI: {n_lat_roi} lat x {n_lon_roi} lon = {n_removed} removed')

# Regional grid: 61 lon x 41 lat = 0.25 deg
r_nlat, r_nlon = 41, 61
r_total = r_nlat * r_nlon
print(f'\\nRegional grid: {r_nlat} lat x {r_nlon} lon = {r_total} nodes')
print(f'Regional resolution: 0.25 deg')
print(f'  lat range: 50..60 => (60-50)/0.25 + 1 = {int((60-50)/0.25)+1} points')
print(f'  lon range: 83..98 => (98-83)/0.25 + 1 = {int((98-83)/0.25)+1} points')

# Total multires nodes
n_global_kept = g_total - n_removed
n_total = n_global_kept + r_total
print(f'\\nMultires total:')
print(f'  Global kept: {n_global_kept}')
print(f'  Regional added: {r_total}')
print(f'  Total nodes: {n_total}')

# Mesh nodes for icosahedron levels 4 and 6
def ico_vertices(level):
    return 10 * (4**level) + 2

m4 = ico_vertices(4)
m6 = ico_vertices(6)
m_total = m6  # finest level
print(f'\\nMesh:')
print(f'  Level 4: {m4} vertices')
print(f'  Level 6: {m6} vertices')

# Decode: each grid node gets 3 mesh->grid edges (CONTAINED mode)
print(f'\\nDecode path:')
print(f'  Each of {n_total} grid nodes gets 3 edges from mesh triangle vertices')
print(f'  Total mesh2grid edges: {n_total * 3}')
print(f'  Regional nodes only: {r_total} * 3 = {r_total * 3} edges')
"


Ran terminal command:  python3 -c 'import numpy as np; g_lats=np.linspace(-90,90,256); g_lons=np.linspace(0,360,512,endpoint=False); m1=(g_lats>=50)&(g_lats<=60); m2=(g_lons>=83)&(g_lons<=98); removed=m1.sum()*m2.sum(); kept=256*512-removed; reg=41*61; print(f"lat_in_roi={m1.sum()} lon_in_roi={m2.sum()} removed={removed} kept={kept} regional={reg} total={kept+reg}")'
Completed with input:  python3 -c 'import numpy as np; g_lats=np.linspace(-90,90,256); g_lons=np.linspace(0,360,512,endpoint=False); m1=(g_lats>=50)&(g_lats<=60); m2=(g_lons>=83)&(g_lons<=98); removed=m1.sum()*m2.sum(); kept=256*512-removed; reg=41*61; print(f"lat_in_roi={m1.sum()} lon_in_roi={m2.sum()} removed={removed} kept={kept} regional={reg} total={kept+reg}")'

Created [](file:///tmp/calc_grid.py)

Ran terminal command:  python3 /tmp/calc_grid.py

Read [](file:///Users/a.s.tabakov/Developer/graphcast-lite/diploma_experiment_results.md#185-185), lines 185 to 200

Searched for regex `jan2023_interp|jan2023_merge|real.*region|real.*0\.25`, 6 results

Starting: *Create prompt file for another AI* (2/2)

Это ключевая находка. Теперь всё встало на место. Пишу промпт-файл: 

Created [](file:///Users/a.s.tabakov/Developer/graphcast-lite/prompt_for_ai.md)

Completed: *Create prompt file for another AI* (2/2)

Промпт создан: prompt_for_ai.md.

А вот мой собственный разбор проблемы и что я думаю.

---

## Корень проблемы: ты учишь модель на пустоте

Главная находка: **обучающие данные для 2,501 мелких точек — это билинейная интерполяция из грубой сетки 0.7°.** В них нет ничего, чего не было бы в 294 грубых точках. Модель учится воспроизводить гладкую интерполяцию, не реальную мелкомасштабную физику.

При этом **реальные ERA5 0.25° для Красноярска за 2010–2020 лежат на диске** (`region_krsk_61x41_19f_2010-2020_025deg`), но **никогда не использовались для обучения**.

---

## Три конкретных идеи, от простой к сложной

### 1. Обучить на реальных 0.25° данных (must-try, быстрая реализация)

Пересобрать dataset с `--mode merge` вместо `--mode interpolate`:
- Для периода 2010–2020 вставить **настоящие** CDS ERA5 0.25° значения в ROI
- Модель увидит реальные мелкомасштабные градиенты (горы, озёра, городские острова тепла)
- Нужно проверить, что scalers (mean/std) пересчитаны с учётом реальных данных

Главный риск — **разрыв на границе ROI**: за пределами ROI данные с разрешением 0.7°, внутри — с 0.25°. Пути митигации:
- **Buffer zone** (2–3°): в переходной зоне blend между coarse и fine
- **Consistency loss**: штрафовать, если прогноз в ROI, усреднённый до 0.7°, сильно отличается от глобального прогноза

### 2. Каскад: GNN → региональный post-processing (средняя сложность)

Вместо одной модели, делаем **два этапа**:
1. **Глобальная GNN** предсказывает на грубой сетке (как раньше, 512×256)
2. **Лёгкая сеть** (MLP или маленький UNet) берёт грубый прогноз в ROI + рельеф + land/sea mask и делает **learned downscaling** до 0.25°

Это работает потому что:
- У-нет на 61×41 — это тривиальная задача, 15 минут обучения (вы уже видели это: `unet_region_krsk` дал 55.7% за 15 минут)
- На вход post-processor получает уже хороший прогноз от GNN + статические поля (рельеф!), которые содержат мелкомасштабную информацию
- Обучение на **реальных** 0.25° данных — быстро, просто, не трогает глобальную модель

### 3. Мелкий mesh в ROI (самая перспективная, но сложная)

Добавить **level 7 icosahedral nodes (~2,300 вершин) только в ROI** в processing graph:
- Processing mesh: level 4–6 глобально + level 7 в ROI
- Encoder: мелкие grid-точки кодируются в мелкие mesh-ноды
- Processor: message passing идёт между всеми mesh-нодами (мелкие ↔ крупные)
- Decoder: мелкие mesh-ноды декодируют в мелкие grid-точки

Это **настоящий multi-resolution**: мелкий mesh может захватить субсеточные паттерны, а cross-level рёбра обеспечивают обмен с глобальной динамикой.

---

## Моя рекомендация: начать с #1, параллельно думать о #2

**Идея #1** — самый быстрый и потенциально самый большой выигрыш. Ты буквально даёшь модели **информацию, которой у неё не было**. Сейчас она учится на пустоте. Нужно:
1. Пересобрать dataset с реальными данными
2. Перезапустить freeze6 обучение

Если хочешь, могу прямо сейчас написать скрипт пересборки dataset с merge-режимом для обучения.
