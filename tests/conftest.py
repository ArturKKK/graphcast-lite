"""Общая обвязка тестов.

Заглушка torch. Постобработка почти вся — это pandas и numpy: отбор признаков,
таблицы поправок, поиск наблюдений назад по времени. Проверять её нужно и там,
где torch не установлен (например, при беглой проверке на ноутбуке), поэтому
подставляем минимальную заглушку, когда настоящего torch нет. Если он есть,
заглушка не ставится и всё идёт через него.
"""
import sys
import types

import pytest


def _install_torch_stub() -> None:
    torch = types.ModuleType("torch")
    torch.from_numpy = lambda x: x
    torch.tensor = lambda x, **kw: x
    torch.float32 = "float32"
    torch.Tensor = object
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _Dataset:  # база для StationCorpusDataset
        pass

    data = types.ModuleType("torch.utils.data")
    data.Dataset = _Dataset
    data.DataLoader = object
    data.WeightedRandomSampler = object
    utils = types.ModuleType("torch.utils")
    utils.data = data
    sys.modules.update({"torch": torch, "torch.utils": utils, "torch.utils.data": data})


try:  # pragma: no cover — зависит от окружения
    import torch  # noqa: F401
except ModuleNotFoundError:
    _install_torch_stub()


@pytest.fixture
def rng():
    import numpy as np
    return np.random.default_rng(20260828)


def load_module(relpath: str, name: str):
    """Загрузить модуль по пути, минуя пакет.

    src/postprocessing/neural/__init__.py тянет за собой models.py, а тот —
    torch.nn целиком. Заглушка такой глубины была бы больше самих тестов и
    проверяла бы уже себя, а не код. Датасет от моделей не зависит, поэтому
    берём его файлом.
    """
    import importlib.util
    import sys
    from pathlib import Path as _P

    root = _P(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
