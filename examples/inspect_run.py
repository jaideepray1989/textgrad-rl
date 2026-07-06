"""Print a compact summary of a TextGrad-RL run directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", default="runs/mac_demo")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"No metrics.csv found at {metrics_path}")
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(f"Run: {run_dir}")
    print(f"Metrics rows: {len(rows)}")
    if rows:
        print(json.dumps(rows[-1], indent=2))
    summary = run_dir / "summary.md"
    if summary.exists():
        print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

