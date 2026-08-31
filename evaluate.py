#!/usr/bin/env python3
"""Evaluate one generated FHE kernel and emit a machine-readable JSON result.

This is intentionally narrower than run_benchmark.py. It runs the harness for a
single case, writes eval_status.json beside the submitted fhe_kernel.py, and
prints one JSON object to stdout. It never writes batch-level report.md or
report.json files.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from fhe_codeeval.harness.evaluator import (
    _DEFAULT_REGISTRY,
    build_payload,
    evaluate,
    write_eval_status,
)


def _write_payload(payload: dict[str, Any], output_path: str | None) -> None:
    text = json.dumps(payload, indent=2, default=str) + "\n"
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text, end="")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one FHE harness evaluation and print pass/fail JSON.")
    parser.add_argument(
        "--case",
        required=True,
        help="Benchmark case id, e.g. packing-oriented-operators/matmul",
    )
    parser.add_argument("--fhe-kernel", required=True, help="Path to fhe_kernel.py")
    parser.add_argument("--method", default="agentic", help="Generation method label")
    parser.add_argument("--model", default="unknown", help="Model label")
    parser.add_argument("--registry", default=_DEFAULT_REGISTRY, help="Path to benchmarks.yaml")
    parser.add_argument("--output", default=None, help="Optional path to also write JSON output")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulator correctness/latency instead of real OpenFHE",
    )
    args = parser.parse_args(argv)

    try:
        result = evaluate(
            args.case,
            args.fhe_kernel,
            method=args.method,
            model=args.model,
            registry_path=args.registry,
            run_real=not args.simulate,
        )
        payload = build_payload(result)
        write_eval_status(payload, args.fhe_kernel)
        _write_payload(payload, args.output)
        return 0 if payload["passed"] else 1
    except Exception:
        payload = {
            "case_id": args.case,
            "passed": False,
            "syntax_correctness": False,
            "functional_correctness": False,
            "reward_hack_detected": False,
            "accuracy": "0/0",
            "latency_ms": None,
            "failures": [{"stage": "evaluator_error", "error": traceback.format_exc()}],
        }
        _write_payload(payload, args.output)
        return 2


if __name__ == "__main__":
    sys.exit(main())
