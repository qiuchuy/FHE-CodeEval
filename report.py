#!/usr/bin/env python3
"""Generate report.json and report.md for one run directory from eval_status.json files."""

from __future__ import annotations

import argparse
import sys

from fhe_codeeval.harness.reporter import print_summary, write_reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan one run directory and generate report.json + report.md from eval_status.json files."
    )
    parser.add_argument(
        "run",
        help="Run directory path or run_id under runs/, e.g. runs/20260421_120000 or 20260421_120000",
    )
    parser.add_argument(
        "--output-dir",
        default="runs",
        help="Base runs directory when the positional argument is a run_id (default: runs)",
    )
    args = parser.parse_args(argv)

    report_json_path, report_md_path, payload = write_reports(args.run, output_dir=args.output_dir)
    print(f"Report JSON → {report_json_path}")
    print(f"Report MD   → {report_md_path}")
    print_summary(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
