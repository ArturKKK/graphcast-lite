"""Агрегатная успешность из сохранённых посрочных ошибок.

Проверка привязана к ОПУБЛИКОВАННЫМ числам: расчёт из `*.npz` обязан
воспроизводить таблицу 3 статьи. Это не формальность — 31.08.2026 пустая клетка
таблицы едва не была заполнена выдуманным значением, а верным способом её
закрытия оказался именно этот расчёт. Если он разойдётся с известными клетками,
доверять восстановленной клетке будет нельзя.
"""
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
RUNS = ROOT / "docs" / "paper" / "runs"

# Ошибка по всей сетке из табл. 3 статьи.
PUBLISHED_GRID = {
    "drop8_krsk_roi_last": 0.169583,
    "roiw10_krsk_roi_last": 0.169707,
    "roiw30_krsk_roi_last": 0.170334,
    "roiw100_krsk_roi_last": 0.171746,
}
# Агрегатная успешность по области из табл. 3.
PUBLISHED_SKILL = {
    "roiw30_krsk_roi_last": 74.85,
    "roiw100_krsk_roi_last": 74.80,
    "chw_krsk_roi": 74.55,
}


def find(tag):
    hits = sorted(glob(str(RUNS / "*" / f"{tag}_samples.npz")))
    if not hits:
        pytest.skip(f"нет сохранённых посрочных ошибок для {tag}")
    return Path(hits[0])


def load(tag):
    from paper_bootstrap_ci import load as _load
    return _load(find(tag))


@pytest.mark.parametrize("tag,expected", sorted(PUBLISHED_GRID.items()))
def test_grid_rmse_matches_the_paper(tag, expected):
    """Ошибка по всей сетке воспроизводит табл. 3 до шестого знака."""
    from paper_bootstrap_ci import dynamic_channels
    d = load(tag)
    ch = dynamic_channels(d["variables"])
    got = float(np.sqrt(d["mse_pred_global"][:, :, ch].mean()))
    assert got == pytest.approx(expected, abs=5e-6), f"{tag}: {got:.6f} против {expected}"


@pytest.mark.parametrize("tag,expected", sorted(PUBLISHED_SKILL.items()))
def test_aggregate_skill_matches_the_paper(tag, expected):
    """Агрегатная успешность воспроизводит табл. 3 с точностью округления."""
    from paper_bootstrap_ci import agg_terms, skill
    d = load(tag)
    pred, base = agg_terms(d, "region", [1, 2, 3, 4])
    assert skill(pred, base) == pytest.approx(expected, abs=0.005)


def test_dynamic_channels_exclude_static_and_forcing():
    """Из агрегата исключены ровно статические поля и форсинг — 27 из 33."""
    from paper_bootstrap_ci import dynamic_channels
    d = load("roiw30_krsk_roi_last")
    ch = dynamic_channels(d["variables"])
    names = [d["variables"][i] for i in ch]
    assert len(ch) == 27, f"динамических каналов {len(ch)}, ожидалось 27"
    for excluded in ("z_surf", "lsm", "sin_hour", "cos_doy"):
        assert excluded not in names


def test_paired_interval_is_narrower_than_unpaired():
    """Парное сравнение на общих сроках даёт более узкий интервал.

    Ради этого сравнение и делается парным: разность двух конфигураций на одном
    и том же сроке гораздо устойчивее, чем разность двух независимых средних.
    """
    from paper_bootstrap_ci import agg_terms, skill_ci, skill_diff_ci
    a = load("roiw30_krsk_roi_last")
    b = load("roiw100_krsk_roi_last")
    pa, ba = agg_terms(a, "region", [1, 2, 3, 4])
    pb, bb = agg_terms(b, "region", [1, 2, 3, 4])
    _, lo_a, hi_a = skill_ci(pa, ba, 20, 400)
    _, lo_d, hi_d = skill_diff_ci(pa, ba, pb, bb, 20, 400)
    assert (hi_d - lo_d) < (hi_a - lo_a), "парный интервал не уже одиночного"


def test_cli_runs_and_reports_significance():
    """Скрипт запускается и печатает вердикт о значимости."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "paper_bootstrap_ci.py"),
         str(find("roiw30_krsk_roi_last")), "--vs", str(find("roiw100_krsk_roi_last")),
         "--var", "aggregate", "--scope", "region", "--reps", "200"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "74.85" in r.stdout and "74.80" in r.stdout
    assert "значимо" in r.stdout
