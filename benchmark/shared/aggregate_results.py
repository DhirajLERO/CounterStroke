"""Merge SCM + Graco metric JSON/txt into summary CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_metrics_txt(path: Path) -> dict[str, str]:
    text = path.read_text() if path.is_file() else ""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("Effectiveness score do("):
            m = re.match(r"Effectiveness score do\((.+?)\): (.+)", line.strip())
            if m:
                parent = m.group(1)
                out[f"effectiveness_do_{parent}"] = m.group(2)
        if line.startswith("FID:"):
            out["fid"] = line.split("mean")[1].strip() if "mean" in line else line
        if "Minimality" in line:
            out["minimality"] = line.strip()
    return out


def _flatten_graco_effectiveness(data: dict) -> dict[str, str]:
    out = {}
    eff = data.get("effectiveness_by_parent") or data.get("effectiveness") or {}
    if isinstance(eff, dict) and eff and "do_parent" not in str(eff):
        first_val = next(iter(eff.values()), None)
        if isinstance(first_val, dict) and all(isinstance(v, (list, tuple)) for v in first_val.values()):
            for parent, scores in eff.items():
                for attr, val in scores.items():
                    out[f"eff_do_{parent}_{attr}"] = str(val)
        else:
            for k, v in eff.items():
                out[f"eff_{k}"] = str(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    variant_names = {"graco_base_o2", "graco_base_o3", "graco50"}
    for metrics_json in args.outputs_root.rglob("graco_metrics.json"):
        data = json.loads(metrics_json.read_text())
        rel = metrics_json.relative_to(args.outputs_root)
        parts = rel.parts
        graph = "default"
        variant = data.get("variant", "")
        if len(parts) >= 2 and parts[-2] in variant_names:
            variant = parts[-2]
            if len(parts) >= 3 and parts[0] != "graco":
                graph = parts[0]
            elif len(parts) >= 3 and parts[0] == "graco" and parts[1] not in variant_names:
                graph = parts[1]
        rows.append(
            {
                "method": "graco",
                "graph": graph,
                "variant": variant,
                **_flatten_graco_effectiveness(data),
                "fid": data.get("fid"),
                "minimality": data.get("minimality"),
            }
        )

    for metrics_txt in args.outputs_root.rglob("metrics.txt"):
        rel = metrics_txt.relative_to(args.outputs_root)
        parts = rel.parts
        if len(parts) < 3:
            continue
        if parts[0] == "scm" and len(parts) >= 4:
            graph, method = parts[1], parts[2]
        else:
            graph, method = parts[0], parts[1]
        parsed = parse_metrics_txt(metrics_txt)
        rows.append({"method": method, "graph": graph, "variant": "", **parsed})

    out_csv = args.out_csv or (args.outputs_root / "summary.csv")
    if not rows:
        print("No metrics found.")
        return
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
