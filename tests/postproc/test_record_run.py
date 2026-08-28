"""Учёт прогонов: строка опыта обновляется, а не дублируется.

Учёт заводился ровно затем, чтобы связь настройки с числом не терялась. Ошибка
здесь тихая: таблица останется правдоподобной, просто в ней будет не то, что
было на самом деле, — и заметить это можно будет только по расхождению со
статьёй, то есть слишком поздно.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "postproc" / "record_run.py"


def make_eval(path: Path, *, t_pp=2.297, t_raw=2.978, w_pp=1.727, w_raw=2.353,
              epoch=14, features=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "eval_per_lead_v2.json").write_text(json.dumps({
        "overall": {"n": 485888, "pp_rmse_t2m": t_pp, "gnn_rmse_t2m": t_raw,
                    "pp_mae_t2m": 1.6, "pp_bias_t2m": -0.03, "gnn_bias_t2m": -0.47,
                    "pp_vec_rmse_wind": w_pp, "gnn_vec_rmse_wind": w_raw,
                    "pp_speed_rmse": 1.2, "gnn_speed_rmse": 1.5},
        "per_lead": {}, "ckpt_epoch": epoch}))
    if features is not None:
        (path.parent / "scalers.json").write_text(
            json.dumps({f"f{i}": [0.0, 1.0] for i in range(features)}))
    return path / "eval_per_lead_v2.json"


def run(ledger: Path, ev: Path, name: str, note: str = ""):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(ledger.parent)}
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--eval-json", str(ev), "--name", name,
         *(["--note", note] if note else [])],
        capture_output=True, text=True, cwd=ROOT,
        env={**dict(__import__("os").environ), "POSTPROC_LEDGER": str(ledger)})
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_gain_is_computed_against_the_raw_forecast(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.md"
    ev = make_eval(tmp_path / "exp" / "eval_test2020", features=60)
    monkeypatch.setenv("POSTPROC_LEDGER", str(ledger))
    run(ledger, ev, "опыт")
    text = ledger.read_text()
    # (2.978 - 2.297) / 2.978 = 22.9 %
    assert "22.9 %" in text
    # (2.353 - 1.727) / 2.353 = 26.6 %
    assert "26.6 %" in text


def test_same_experiment_updates_its_row(tmp_path, monkeypatch):
    """Перепроверил тот же опыт — строка обновилась, а не задвоилась."""
    ledger = tmp_path / "ledger.md"
    monkeypatch.setenv("POSTPROC_LEDGER", str(ledger))
    ev = make_eval(tmp_path / "exp" / "eval_test2020", t_pp=2.400, features=60)
    run(ledger, ev, "опыт")
    ev = make_eval(tmp_path / "exp" / "eval_test2020", t_pp=2.100, features=60)
    run(ledger, ev, "опыт")
    lines = [l for l in ledger.read_text().splitlines() if l.startswith("| `опыт`")]
    assert len(lines) == 1, "строка задвоилась"
    assert "2.100" in lines[0] and "2.400" not in lines[0]


def test_different_experiments_get_their_own_rows(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.md"
    monkeypatch.setenv("POSTPROC_LEDGER", str(ledger))
    for name, t in (("первый", 2.30), ("второй", 2.20)):
        ev = make_eval(tmp_path / name / "eval_test2020", t_pp=t, features=60)
        run(ledger, ev, name)
    rows = [l for l in ledger.read_text().splitlines() if l.startswith("| `")]
    assert len(rows) == 2


def test_feature_count_comes_from_the_checkpoint(tmp_path, monkeypatch):
    """Число признаков читается из нормировок, а не задаётся руками.

    Эта графа в первый же день поймала подвох: у настройки с жребием 43 их
    оказалось 62 против 60 у основной, и сравнивать их напрямую было нельзя.
    """
    ledger = tmp_path / "ledger.md"
    monkeypatch.setenv("POSTPROC_LEDGER", str(ledger))
    ev = make_eval(tmp_path / "exp" / "eval_test2020", features=73)
    run(ledger, ev, "опыт")
    assert "| 73 |" in ledger.read_text()


def test_missing_scalers_do_not_break_the_record(tmp_path, monkeypatch):
    """Нет нормировок — в графе вопрос, но строка всё равно пишется."""
    ledger = tmp_path / "ledger.md"
    monkeypatch.setenv("POSTPROC_LEDGER", str(ledger))
    ev = make_eval(tmp_path / "exp" / "eval_test2020")
    run(ledger, ev, "опыт")
    assert "| ? |" in ledger.read_text()


def test_header_is_written_once(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger.md"
    monkeypatch.setenv("POSTPROC_LEDGER", str(ledger))
    for name in ("a", "b", "c"):
        ev = make_eval(tmp_path / name / "eval_test2020", features=60)
        run(ledger, ev, name)
    assert ledger.read_text().count("| опыт | дата |") == 1
