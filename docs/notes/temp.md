# === 1. Распаковать глобальный датасет ===
cd /data/datasets
zstd -d dataset_512x256.tar.zst -o dataset_512x256.tar && tar xf dataset_512x256.tar

# Переместить и переименовать (внутри будет data/datasets/wb2_512x256_19f_ar/)
mv data/datasets/wb2_512x256_19f_ar /data/datasets/global_512x256_19f_2010-2021_07deg
rm -rf data  # пустая обёртка

# === 2. Распаковать региональный датасет ===
tar xzf region_krsk_61x41_19f_2010-2020_025deg.tar.gz

# === 3. Удалить архивы (освободить место) ===
rm dataset_512x256.tar.zst dataset_512x256.tar region_krsk_61x41_19f_2010-2020_025deg.tar.gz

# === 4. Проверить ===
ls global_512x256_19f_2010-2021_07deg/
ls region_krsk_61x41_19f_2010-2020_025deg/
# Должны быть: data.npy, dataset_info.json, coords.npz, scalers.npz, variables.json

# === 5. Симлинки в workdir ===
ln -s /data/datasets/global_512x256_19f_2010-2021_07deg /workdir/graphcast-lite/data/datasets/global_512x256_19f_2010-2021_07deg
ln -s /data/datasets/region_krsk_61x41_19f_2010-2020_025deg /workdir/graphcast-lite/data/datasets/region_krsk_61x41_19f_2010-2020_025deg

# === 6. Собрать мультирез датасет ===
cd /workdir/graphcast-lite
python scripts/build_multires_dataset.py \
    --global-dir data/datasets/global_512x256_19f_2010-2021_07deg \
    --region-coords data/datasets/region_krsk_61x41_19f_2010-2020_025deg/coords.npz \
    --roi 50 60 83 98 \
    --mode interpolate \
    --out-dir /data/datasets/multires_krsk_19f

ln -s /data/datasets/multires_krsk_19f data/datasets/multires_krsk_19f

# === 7. Fine-tune глобальной модели ===
python -m src.main experiments/multires_krsk_19f \
    --pretrained experiments/wb2_512x256_19f_ar_v2/best_model.pth

# === 8. Инференс ===
python scripts/predict.py experiments/multires_krsk_19f \
    --prune-mesh --ar-steps 4 --per-channel --max-samples 200

python scripts/refresh_presentation_artifacts.py \
  --main-data /data/datasets/multires_krsk_19f \
  --jan-data /data/datasets/multires_krsk_19f_jan2023_interp \
  --wrf-json /workdir/graphcast-list/wrf_d03_jan2023.json \
  --out-dir results/presentation_refresh_9day \
  --artifact-store-dir /data/graphcast-lite/presentation_cache/presentation_refresh_9day/artifacts \
  --ar-steps 36 \
  --artifact-samples 4 \
  --global-exp experiments/wb2_64x32_ar_15f_4obs_4pred \
  --global-data data/datasets/wb2_64x32_zq_15f_4obs_4pred \
  --global-ar-steps 12 \
  --global-max-samples 200