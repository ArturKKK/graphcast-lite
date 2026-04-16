#!/usr/bin/env python3
"""Parse DA experiment logs and produce structured results tables."""
import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict

def parse_log(logpath):
    """Extract metrics from a single experiment log."""
    text = logpath.read_text()
    result = {"name": logpath.stem, "horizons": {}}

    # Overall metrics
    m = re.search(r"Overall: MSE=([\d.]+) \| RMSE=([\d.]+) \| MAE=([\d.]+)", text)
    if m:
        result["overall_mse"] = float(m.group(1))
        result["overall_rmse"] = float(m.group(2))
        result["overall_mae"] = float(m.group(3))

    # Full horizons
    for m in re.finditer(
        r"\+(\d+)h: RMSE=([\d.]+) \| base=([\d.]+) \| skill=([\d.]+)% \| ACC=([\d.]+) \(base ([\d.]+)\)",
        text
    ):
        h = int(m.group(1))
        result["horizons"][h] = {
            "rmse": float(m.group(2)),
            "base_rmse": float(m.group(3)),
            "skill": float(m.group(4)),
            "acc": float(m.group(5)),
            "base_acc": float(m.group(6)),
        }

    # Region metrics (second block of +XYh lines after "Region" heading or RMSE=... line)
    region_block = re.findall(
        r"(?:Region|RMSE=[\d.]+.*?skill=[\d.]+%).*?"
        r"(\+\d+h:.*?)(?:\n\n|\Z)",
        text, re.DOTALL
    )
    result["region"] = {}
    # Find lines with 4-space indent (region per-horizon)
    for m in re.finditer(
        r"    \+(\d+)h: RMSE=([\d.]+) \| base=([\d.]+) \| skill=([\d.]+)% \| ACC=([\d.]+) \(base ([\d.]+)\)",
        text
    ):
        h = int(m.group(1))
        result["region"][h] = {
            "rmse": float(m.group(2)),
            "base_rmse": float(m.group(3)),
            "skill": float(m.group(4)),
            "acc": float(m.group(5)),
            "base_acc": float(m.group(6)),
        }

    # Region summary line
    m = re.search(r"RMSE=([\d.]+) \| base=([\d.]+) \| skill=([\d.]+)%\n", text)
    if m:
        result["region_summary"] = {
            "rmse": float(m.group(1)),
            "base_rmse": float(m.group(2)),
            "skill": float(m.group(3)),
        }

    # Per-channel RMSE (physical units)
    result["per_channel"] = {}
    for m in re.finditer(r"\[per-ch\]\s+(\S+)\s+.*?region_rmse=([\d.]+)", text):
        result["per_channel"][m.group(1)] = float(m.group(2))
    # Alternative format
    for m in re.finditer(r"(\w+(?:@\d+)?)\s+\|\s+[\d.]+\s+\|\s+([\d.]+)\s+\|", text):
        if m.group(1) not in result["per_channel"]:
            result["per_channel"][m.group(1)] = float(m.group(2))

    return result


def _parse_corr_km(s):
    """Parse corr_len value from name fragment like '100' or '100km'."""
    return int(s.replace("km", ""))

def make_oi_table(results, density_prefix, metric_key="skill"):
    """Build corr_len × sigma table for OI experiments."""
    corr_lens = sorted(set(
        _parse_corr_km(r["name"].split("_c")[1].split("_")[0])
        for r in results if r["name"].startswith(density_prefix + "_c")
    ))
    sigmas = sorted(set(
        float(r["name"].split("_s")[1])
        for r in results if r["name"].startswith(density_prefix + "_c")
    ))

    lookup = {}
    for r in results:
        if not r["name"].startswith(density_prefix + "_c"):
            continue
        parts = r["name"].split("_")
        c = _parse_corr_km(parts[1][1:])
        s = float(parts[2][1:])
        if r.get("region") and 6 in r["region"]:
            lookup[(c, s)] = r["region"][6].get(metric_key, "—")
        elif r.get("region_summary"):
            lookup[(c, s)] = r["region_summary"].get(metric_key, "—")

    # Format table
    header = f"| corr_len \\ σ | " + " | ".join(f"σ={s}" for s in sigmas) + " |"
    sep = "|---|" + "|".join("---" for _ in sigmas) + "|"
    rows = []
    for c in corr_lens:
        vals = []
        for s in sigmas:
            v = lookup.get((c, s), "—")
            if isinstance(v, float):
                vals.append(f"{v:.2f}%")
            else:
                vals.append(str(v))
        rows.append(f"| {c} km | " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + rows)


def make_nudging_table(results, density_prefix):
    """Build alpha × mode table for nudging experiments."""
    lines = []
    for r in sorted(results, key=lambda x: x["name"]):
        if not r["name"].startswith(density_prefix):
            continue
        skill_6h = "—"
        if r.get("region") and 6 in r["region"]:
            skill_6h = f"{r['region'][6]['skill']:.2f}%"
        lines.append(f"| {r['name']} | {skill_6h} |")

    header = "| Config | Region +6h Skill |"
    sep = "|---|---|"
    return "\n".join([header, sep] + lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_da_results.py <log_dir> [<log_dir2>]")
        sys.exit(1)

    for log_dir_path in sys.argv[1:]:
        log_dir = Path(log_dir_path)
        if not log_dir.exists():
            print(f"Directory not found: {log_dir}")
            continue

        logs = sorted(log_dir.glob("*.log"))
        if not logs:
            print(f"No .log files in {log_dir}")
            continue

        results = []
        for lp in logs:
            try:
                results.append(parse_log(lp))
            except Exception as e:
                print(f"WARN: failed to parse {lp.name}: {e}", file=sys.stderr)

        tag = log_dir.name
        print(f"\n{'='*60}")
        print(f"  {tag}: {len(results)} experiments")
        print(f"{'='*60}\n")

        # Baseline
        baseline = [r for r in results if r["name"] == "baseline"]
        if baseline:
            b = baseline[0]
            print("## Baseline")
            if b.get("region"):
                for h in sorted(b["region"]):
                    rm = b["region"][h]
                    print(f"  +{h}h: Skill={rm['skill']:.2f}% ACC={rm['acc']:.4f}")
            print()

        # OI tables
        for density, prefix in [("10%", "oi10"), ("1%", "oi1")]:
            matching = [r for r in results if r["name"].startswith(prefix + "_c")]
            if matching:
                print(f"## OI {density} — Region +6h Skill")
                print(make_oi_table(results, prefix))
                print()

        # Nudging tables
        for density, prefix in [("10%", "nudg10"), ("1%", "nudg1")]:
            matching = [r for r in results if r["name"].startswith(prefix)]
            if matching:
                print(f"## Nudging {density}")
                print(make_nudging_table(results, prefix))
                print()

        # Variable groups
        vargroup = [r for r in results if r["name"].startswith("vargroup_")]
        if vargroup:
            print("## Variable Groups (OI 10%, c=100, σ=0.5)")
            print("| Group | Region +6h Skill |")
            print("|---|---|")
            for r in vargroup:
                skill = "—"
                if r.get("region") and 6 in r["region"]:
                    skill = f"{r['region'][6]['skill']:.2f}%"
                print(f"| {r['name'].replace('vargroup_', '').replace('_oi10', '')} | {skill} |")
            print()

        # Best configs per-horizon detail
        best_oi10 = max(
            [r for r in results if r["name"].startswith("oi10_") and r.get("region") and 6 in r["region"]],
            key=lambda r: r["region"][6]["skill"],
            default=None
        )
        if best_oi10:
            print(f"## Best OI 10%: {best_oi10['name']}")
            print("| Horizon | Region RMSE | Skill | ACC |")
            print("|---|---|---|---|")
            for h in sorted(best_oi10["region"]):
                rm = best_oi10["region"][h]
                print(f"| +{h}h | {rm['rmse']:.4f} | {rm['skill']:.2f}% | {rm['acc']:.4f} |")
            print()

        best_oi1 = max(
            [r for r in results if r["name"].startswith("oi1_") and r.get("region") and 6 in r["region"]],
            key=lambda r: r["region"][6]["skill"],
            default=None
        )
        if best_oi1:
            print(f"## Best OI 1%: {best_oi1['name']}")
            print("| Horizon | Region RMSE | Skill | ACC |")
            print("|---|---|---|---|")
            for h in sorted(best_oi1["region"]):
                rm = best_oi1["region"][h]
                print(f"| +{h}h | {rm['rmse']:.4f} | {rm['skill']:.2f}% | {rm['acc']:.4f} |")
            print()

        # Dump JSON for further analysis
        json_path = log_dir / "parsed_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Parsed JSON saved to {json_path}]")


if __name__ == "__main__":
    main()
