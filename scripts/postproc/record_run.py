#!/usr/bin/env python3
"""Заносит итог прогона постпроцессора в общий учёт.

Зачем. За один день 28.08.2026 обучилось десять настроек, и их числа жили только
в логах виртуалки: чтобы сравнить две, приходилось листать master-лог и сличать
строки «Overall» на глаз. Через неделю связь между настройкой и числом теряется —
ровно так уже вышло с ранними версиями основной модели (см. MODELS.md).

Что пишется: имя опыта, дата, коммит, число признаков, годы обучения, эпоха
лучшего чекпойнта и метрики на проверке — вместе с выигрышем против сырого
прогноза, который иначе каждый раз считают в уме.

Строка опыта обновляется на месте, а не дублируется: перепроверил тот же
чекпойнт — обновилась дата и метрики, а не появилась вторая запись.

Запуск:
    python3 scripts/postproc/record_run.py --eval-json ПУТЬ --name ИМЯ \
        [--note "чем отличается"]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "docs" / "postproc_runs.md"

HEADER = """# Учёт прогонов постобработки

Заполняется автоматически из `scripts/postproc/record_run.py`, который зовут
раннеры после каждой проверки. Руками сюда писать не нужно.

Проверка везде — 2020 год, 485 888 строк (те, где есть все три наблюдения).
Сырой прогноз на ней: t2m 2,978 °C, ветер по вектору 2,353 м/с.

| опыт | дата | коммит | призн. | эпоха | t2m | выигрыш | ветер | выигрыш | чем отличается |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
"""


def git_commit() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10)
        mark = "*" if dirty.stdout.strip() else ""
        return h.stdout.strip() + mark
    except Exception:
        return "?"


def read_features(ckpt_dir: Path) -> str:
    """Сколько признаков у модели — из чекпойнта, не из догадок."""
    for name in ("scalers.json",):
        p = ckpt_dir / name
        if p.exists():
            try:
                return str(len(json.loads(p.read_text())))
            except Exception:
                pass
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--name", required=True, help="имя опыта, оно же ключ строки")
    ap.add_argument("--note", default="", help="чем настройка отличается")
    a = ap.parse_args()

    ev = json.loads(Path(a.eval_json).read_text())
    o = ev["overall"]
    t_pp, t_raw = o["pp_rmse_t2m"], o["gnn_rmse_t2m"]
    w_pp, w_raw = o["pp_vec_rmse_wind"], o["gnn_vec_rmse_wind"]
    gain = lambda raw, pp: (raw - pp) / raw * 100 if raw else float("nan")

    row = ("| `{name}` | {date} | `{commit}` | {feat} | {epoch} | "
           "{t:.3f} | {tg:.1f} % | {w:.3f} | {wg:.1f} % | {note} |").format(
        name=a.name, date=datetime.now().strftime("%d.%m.%Y %H:%M"),
        commit=git_commit(), feat=read_features(Path(a.eval_json).parent.parent),
        epoch=ev.get("ckpt_epoch", "?"), t=t_pp, tg=gain(t_raw, t_pp),
        w=w_pp, wg=gain(w_raw, w_pp), note=a.note or "—")

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    lines = LEDGER.read_text().splitlines() if LEDGER.exists() else HEADER.splitlines()
    key = f"| `{a.name}` |"
    for i, line in enumerate(lines):
        if line.startswith(key):
            lines[i] = row
            break
    else:
        lines.append(row)
    LEDGER.write_text("\n".join(lines) + "\n")
    print(f"[учёт] {a.name}: t2m {t_pp:.3f} ({gain(t_raw, t_pp):.1f} %), "
          f"ветер {w_pp:.3f} ({gain(w_raw, w_pp):.1f} %) -> {LEDGER.name}", flush=True)


if __name__ == "__main__":
    main()
