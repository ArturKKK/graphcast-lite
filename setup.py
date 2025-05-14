# setup.py  (лежит в корне репозитория, вы его уже видите)
from setuptools import setup, find_packages

setup(
    name="graphcast_lite",           # pip install graphcast-lite
    version="0.1.0",
    package_dir={"": "src"},         # ищем пакеты только в src/
    packages=find_packages("src"),   # graphcast_lite, graphcast_lite.data, …
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0,<2.3",
        "torch_geometric==2.5.3",
        "numpy==1.26.4",
        "xarray==2023.12.0",
        "zarr==2.11.3",
        "dask==2022.6.0",
        "fsspec==2022.5.0",
        "gcsfs==2022.5.0",
        "pydantic==2.7.3",
        "trimesh==4.4.0",
        "cartopy==0.23.0",
        "wandb>=0.16",
    ],
)
