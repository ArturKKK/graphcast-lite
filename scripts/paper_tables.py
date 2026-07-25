#!/usr/bin/env python3
"""Парсит логи прогонов predict.py и собирает таблицы для статьи (markdown).

Читает *.log из каталога (скачанные с VM), достаёт:
  - провенанс (коммит, чекпойнт, команда) — чтобы каждое число было прослеживаемо;
  - агрегатные RMSE/Skill/корреляцию по региону и глобально;
  - per-horizon Skill и per-horizon RMSE по каналам (физические единицы).

Использование:
    python scripts/paper_tables.py vm_backup/paper_results/vm3_run1 [--out docs/paper/tables.md]
"""
import argparse
import re
from pathlib import Path

HORIZ = ["+6h", "+12h", "+18h", "+24h"]


def parse_log(p: Path) -> dict:
    txt = p.read_text(errors="replace")
    d = {"file": p.name, "tag": p.stem}

    # --- провенанс ---
    for key, pat in [
        ("commit", r"^# git commit:\s*(\S+)"),
        ("started", r"^# started:\s*(\S+)"),
        ("dataset", r"^# dataset:\s*(\S+)"),
        ("command", r"^#\s+(python -u scripts/predict\.py .+)$"),
    ]:
        m = re.search(pat, txt, re.M)
        if m:
            d[key] = m.group(1).strip()
    ck = re.findall(r"^#\s+ckpt(?:\(--ckpt\))?:\s*([0-9a-f]{32})\s+\S*\s*(\S+)?", txt, re.M)
    if ck:
        d["ckpts"] = ck

    # --- сколько сэмплов реально прогнано (test_only) ---
    m = re.search(r"\[ChunkDataset\] test_only:\s*(\d+) samples", txt)
    if m:
        d["n_samples"] = m.group(1)

    # --- глобальный агрегат ---
    mg = re.search(r"^Overall:.*?RMSE=([\d.]+).*?\n^Baseline:.*?RMSE=([\d.]+).*?\n^Skill:\s*([\d.-]+)%",
                   txt, re.M | re.S)
    if mg:
        d["global_rmse"], d["global_base"], d["global_skill"] = mg.groups()

    # --- РЕГИОНАЛЬНЫЙ агрегат (блок '--- Region ... ---') ---
    mreg = re.search(
        r"^--- Region .*?\((\d+) nodes\) ---\s*\n"
        r"RMSE=([\d.]+)\s*\|\s*base=([\d.]+)\s*\|\s*skill=([\d.-]+)%\s*\n"
        r"ACC=([\d.]+)\s*\|\s*base=([\d.]+)", txt, re.M)
    if mreg:
        (d["region_nodes"], d["region_rmse_norm"], d["region_base"],
         d["region_skill"], d["region_corr"], d["region_corr_base"]) = mreg.groups()

    # --- per-horizon региона (строки внутри 'Per-horizon (region):') ---
    mblock = re.search(r"Per-horizon \(region\):\s*\n((?:\s+\+\d+h:.*\n)+)", txt)
    if mblock:
        d["per_horizon"] = re.findall(
            r"\+(\d+)h:\s*RMSE=([\d.]+)\s*\|\s*base=([\d.]+)\s*\|\s*skill=([\d.-]+)%\s*\|\s*ACC=([\d.]+)",
            mblock.group(1))

    # --- per-horizon skill (последний блок = региональный, если есть --region) ---
    ph = re.findall(r"\+(\d+)h:\s*RMSE=([\d.]+)\s*\|\s*base=([\d.]+)\s*\|\s*skill=([\d.-]+)%", txt)
    if ph:
        # берём последние 4 (регион)
        d["per_horizon"] = ph[-4:] if len(ph) >= 4 else ph

    # --- per-horizon per-channel в физических единицах ---
    # приоритет — региональная таблица; если её нет, берём глобальную
    blocks = re.findall(
        r"Per-horizon per-channel RMSE — REGION[^\n]*\n(.*?)(?:\n\s*\n|\n\s+Per-channel)",
        txt, re.S)
    if not blocks:
        blocks = re.findall(
            r"Per-horizon per-channel RMSE \(physical units\):\n(.*?)(?:\n\s*\n|\nPer-channel)",
            txt, re.S)
    if blocks:
        rows = {}
        for line in blocks[-1].splitlines():
            parts = line.split()
            if len(parts) >= 6 and parts[0] != "var":
                rows[parts[0]] = (parts[1], parts[2:6])
        d["channels"] = rows
    return d


def fmt_tables(logs: list[dict]) -> str:
    out = ["# Таблицы для статьи (сгенерировано из логов прогонов)", ""]

    # ---------- Табл. 1: сводка по моделям ----------
    out += ["## Табл. 1. Сводка по конфигурациям (регион, полный тест)", "",
            "| прогон | N сэмплов | Skill региона | t2m +6ч | +12ч | +18ч | +24ч |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for d in logs:
        t2m = d.get("channels", {}).get("t2m", ("", ["—"] * 4))[1]
        t2m = [v.replace("°C", "") for v in t2m]
        out.append("| {} | {} | {} | {} |".format(
            d["tag"], d.get("n_samples", "?"),
            (d.get("region_skill", "?") + "%") if d.get("region_skill") else "?",
            " | ".join(t2m)))
    out.append("")

    # ---------- Табл. 2: per-horizon Skill ----------
    out += ["## Табл. 2. Skill по горизонтам (регион)", "",
            "| прогон | " + " | ".join(HORIZ) + " |", "|---|" + "---:|" * 4]
    for d in logs:
        ph = d.get("per_horizon")
        if not ph:
            continue
        cells = [f"{s}%" for (_h, _r, _b, s) in ph]
        out.append(f"| {d['tag']} | " + " | ".join(cells) + " |")
    out.append("")

    # ---------- Табл. 3: физические RMSE по каналам ----------
    out += ["## Табл. 3. RMSE по каналам, физические единицы (регион, по горизонтам)", ""]
    for d in logs:
        ch = d.get("channels")
        if not ch:
            continue
        out += [f"### {d['tag']}", "",
                "| переменная | единица | " + " | ".join(HORIZ) + " |",
                "|---|---|" + "---:|" * 4]
        for var, (unit, vals) in ch.items():
            out.append(f"| {var} | {unit} | " + " | ".join(v.replace('°C', '') for v in vals) + " |")
        out.append("")

    # ---------- Провенанс ----------
    out += ["## Провенанс прогонов", "",
            "| прогон | коммит | запущен | чекпойнт (md5) |", "|---|---|---|---|"]
    for d in logs:
        cks = d.get("ckpts", [])
        ckstr = ", ".join(f"`{h[:8]}`" for h, _ in cks[:2]) or "—"
        out.append(f"| {d['tag']} | `{d.get('commit','?')}` | {d.get('started','?')} | {ckstr} |")
    out += ["", "Команды запуска:", "```"]
    for d in logs:
        if d.get("command"):
            out.append(f"# {d['tag']}\n{d['command']}\n")
    out.append("```")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = sorted(Path(a.log_dir).glob("*.log"))
    files = [f for f in files if "master" not in f.name]
    if not files:
        raise SystemExit(f"нет .log в {a.log_dir}")
    logs = [parse_log(f) for f in files]
    md = fmt_tables(logs)
    if a.out:
        Path(a.out).write_text(md)
        print(f"→ {a.out} ({len(files)} логов)")
    else:
        print(md)


if __name__ == "__main__":
    main()
