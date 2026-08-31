"""
Run-directory reporting based on per-case eval_status.json artifacts.

Writes:
  runs/{run_id}/report.json  — machine-readable aggregate of eval_status payloads
  runs/{run_id}/report.md    — human-readable summary table
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _resolve_run_dir(run: str | Path, output_dir: str = "runs") -> Path:
    candidate = Path(run)
    if candidate.exists() and candidate.is_dir():
        return candidate
    return Path(output_dir) / str(run)


def _infer_sample_id(status_path: Path, run_dir: Path) -> str | None:
    try:
        rel_parent = status_path.parent.relative_to(run_dir)
    except ValueError:
        return None

    parts = rel_parent.parts
    if len(parts) >= 2 and parts[-1].isdigit():
        return parts[-1]
    return None


def _accuracy_ratio(value: Any) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    passed_str, total_str = value.split("/", 1)
    try:
        passed = int(passed_str)
        total = int(total_str)
    except ValueError:
        return None
    if total <= 0:
        return None
    return passed / total


def _sample_sort_key(value: Any) -> tuple[int, Any]:
    if value is None:
        return (0, "")
    text = str(value)
    if text.isdigit():
        return (1, int(text))
    return (2, text)


def _unbiased_pass_at_k(n: int, correct: int, k: int) -> float | None:
    """Return the unbiased Pass@k estimate for one case."""
    if n < k or k < 1:
        return None
    if correct == 0:
        return 0.0
    if n - correct < k:
        return 1.0
    return 1.0 - (math.comb(n - correct, k) / math.comb(n, k))


def _mean_pass_at_k(by_case: dict[str, list[dict[str, Any]]], k: int) -> float | None:
    if not by_case or any(len(rows) < k for rows in by_case.values()):
        return None
    scores = []
    for rows in by_case.values():
        score = _unbiased_pass_at_k(
            len(rows),
            sum(1 for row in rows if row.get("passed")),
            k,
        )
        if score is not None:
            scores.append(score)
    return (sum(scores) / len(scores)) if scores else None


def collect_eval_statuses(run: str | Path, output_dir: str = "runs") -> list[dict[str, Any]]:
    """Load all eval_status.json payloads under one run directory."""
    run_dir = _resolve_run_dir(run, output_dir=output_dir)
    statuses: list[dict[str, Any]] = []

    if not run_dir.exists():
        return statuses

    for status_path in sorted(run_dir.rglob("eval_status.json")):
        payload = json.loads(status_path.read_text())
        rel_artifact_dir = status_path.parent.relative_to(run_dir).as_posix()
        rel_status_path = status_path.relative_to(run_dir).as_posix()

        entry = dict(payload)
        entry["artifact_dir"] = rel_artifact_dir
        entry["status_path"] = rel_status_path
        entry.setdefault("sample_id", _infer_sample_id(status_path, run_dir))
        entry["accuracy_ratio"] = _accuracy_ratio(entry.get("accuracy"))
        entry["failure_count"] = len(entry.get("failures", []))
        statuses.append(entry)

    statuses.sort(
        key=lambda row: (
            str(row.get("case_id", "")),
            _sample_sort_key(row.get("sample_id")),
            str(row.get("artifact_dir", "")),
        )
    )
    return statuses


def build_report_payload(run: str | Path, output_dir: str = "runs") -> dict[str, Any]:
    """Build the aggregate JSON payload for one run directory."""
    run_dir = _resolve_run_dir(run, output_dir=output_dir)
    statuses = collect_eval_statuses(run_dir)

    total = len(statuses)
    passed = sum(1 for row in statuses if row.get("passed"))
    syntax = sum(1 for row in statuses if row.get("syntax_correctness"))
    functional = sum(1 for row in statuses if row.get("functional_correctness"))
    reward_hack = sum(1 for row in statuses if row.get("reward_hack_detected"))
    latencies = [row["latency_ms"] for row in statuses if row.get("latency_ms") is not None]

    reward_hack_tag_counts: dict[str, int] = defaultdict(int)
    for row in statuses:
        for tag in row.get("reward_hack_tags", []):
            reward_hack_tag_counts[tag] += 1

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in statuses:
        by_case[str(row.get("case_id", ""))].append(row)

    pass_at_1 = _mean_pass_at_k(by_case, 1)
    pass_at_5 = _mean_pass_at_k(by_case, 5)

    case_summary: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        rows = by_case[case_id]
        ratios = [row["accuracy_ratio"] for row in rows if row.get("accuracy_ratio") is not None]
        case_latencies = [row["latency_ms"] for row in rows if row.get("latency_ms") is not None]
        case_summary.append(
            {
                "case_id": case_id,
                "entries": len(rows),
                "passed": sum(1 for row in rows if row.get("passed")),
                "syntax_correctness": sum(1 for row in rows if row.get("syntax_correctness")),
                "functional_correctness": sum(1 for row in rows if row.get("functional_correctness")),
                "reward_hack_detected": sum(1 for row in rows if row.get("reward_hack_detected")),
                "avg_accuracy_ratio": (sum(ratios) / len(ratios)) if ratios else None,
                "avg_latency_ms": (sum(case_latencies) / len(case_latencies)) if case_latencies else None,
            }
        )

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "summary": {
            "evaluated_entries": total,
            "models": sorted({str(row.get("model", "unknown")) for row in statuses}),
            "methods": sorted({str(row.get("method", "unknown")) for row in statuses}),
            "passed": passed,
            "failed": total - passed,
            "syntax_correctness": syntax,
            "functional_correctness": functional,
            "reward_hack_detected": reward_hack,
            "reward_hack_tag_counts": dict(reward_hack_tag_counts),
            "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
            "pass_at_1": pass_at_1,
            "pass_at_5": pass_at_5,
        },
        "case_summary": case_summary,
        "cases": statuses,
    }


def _render_bool(value: Any) -> str:
    return "✓" if value else "✗"


def render_report_markdown(payload: dict[str, Any]) -> str:
    """Render report.md content from a report payload."""
    run_id = payload["run_id"]
    run_dir = payload["run_dir"]
    summary = payload["summary"]
    cases = payload["cases"]
    case_summary = payload["case_summary"]

    lines: list[str] = []
    lines.append(f"# FHE Benchmark Report — Run `{run_id}`")
    lines.append("")
    lines.append(f"Generated from `eval_status.json` files under `{run_dir}`.")
    lines.append("")
    lines.append(f"Model(s): {', '.join(summary.get('models', [])) or 'unknown'}")
    lines.append(f"Method(s): {', '.join(summary.get('methods', [])) or 'unknown'}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Evaluated | Passed | Failed | Pass@1 | Pass@5 | Syntax OK | "
        "Functional OK | Reward Hacks | Avg Latency (ms) |"
    )
    lines.append(
        "|-----------|--------|--------|--------|--------|-----------|"
        "---------------|--------------|------------------|"
    )
    avg_latency = summary["avg_latency_ms"]
    avg_latency_str = f"{avg_latency:.1f}" if avg_latency is not None else "—"
    pass_at_1 = summary.get("pass_at_1")
    pass_at_5 = summary.get("pass_at_5")
    pass_at_1_str = f"{100 * pass_at_1:.1f}%" if pass_at_1 is not None else "—"
    pass_at_5_str = f"{100 * pass_at_5:.1f}%" if pass_at_5 is not None else "—"
    lines.append(
        f"| {summary['evaluated_entries']} | {summary['passed']} | {summary['failed']} | "
        f"{pass_at_1_str} | {pass_at_5_str} | "
        f"{summary['syntax_correctness']} | {summary['functional_correctness']} | "
        f"{summary['reward_hack_detected']} | {avg_latency_str} |"
    )
    lines.append("")

    tag_counts = summary.get("reward_hack_tag_counts", {})
    if tag_counts:
        lines.append("### Reward Hack Breakdown")
        lines.append("")
        lines.append("| Tag | Count |")
        lines.append("|-----|-------|")
        for tag, count in sorted(tag_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{tag}` | {count} |")
        lines.append("")

    lines.append("## Per-Case Summary")
    lines.append("")
    if case_summary:
        lines.append(
            "| Case | Entries | Passed | Syntax OK | Functional OK | Reward Hacks | Avg Accuracy | Avg Latency (ms) |"
        )
        lines.append(
            "|------|---------|--------|-----------|---------------|--------------|--------------|------------------|"
        )
        for row in case_summary:
            avg_accuracy = row["avg_accuracy_ratio"]
            avg_accuracy_str = f"{avg_accuracy:.3f}" if avg_accuracy is not None else "—"
            avg_latency = row["avg_latency_ms"]
            avg_latency_str = f"{avg_latency:.1f}" if avg_latency is not None else "—"
            lines.append(
                f"| `{row['case_id']}` | {row['entries']} | {row['passed']} | "
                f"{row['syntax_correctness']} | {row['functional_correctness']} | "
                f"{row['reward_hack_detected']} | {avg_accuracy_str} | {avg_latency_str} |"
            )
    else:
        lines.append("No `eval_status.json` files found.")
    lines.append("")

    lines.append("## Case Results")
    lines.append("")
    if cases:
        show_samples = any(row.get("sample_id") is not None for row in cases)
        if show_samples:
            lines.append(
                "| Case | Sample | Passed | Syntax | Functional | Reward Hack | "
                "Accuracy | Latency (ms) | Failures | Artifact Dir |"
            )
            lines.append(
                "|------|--------|--------|--------|------------|-------------|----------|--------------|----------|--------------|"
            )
        else:
            lines.append(
                "| Case | Passed | Syntax | Functional | Reward Hack | Accuracy | "
                "Latency (ms) | Failures | Artifact Dir |"
            )
            lines.append(
                "|------|--------|--------|------------|-------------|----------|--------------|----------|--------------|"
            )

        for row in cases:
            latency = row.get("latency_ms")
            latency_str = f"{latency:.1f}" if latency is not None else "—"
            reward_hack_str = _render_bool(row.get("reward_hack_detected"))
            if show_samples:
                lines.append(
                    f"| `{row.get('case_id', '')}` | {row.get('sample_id') or '—'} | "
                    f"{_render_bool(row.get('passed'))} | {_render_bool(row.get('syntax_correctness'))} | "
                    f"{_render_bool(row.get('functional_correctness'))} | {reward_hack_str} | "
                    f"{row.get('accuracy', '—')} | {latency_str} | {row.get('failure_count', 0)} | "
                    f"`{row.get('artifact_dir', '')}` |"
                )
            else:
                lines.append(
                    f"| `{row.get('case_id', '')}` | {_render_bool(row.get('passed'))} | "
                    f"{_render_bool(row.get('syntax_correctness'))} | "
                    f"{_render_bool(row.get('functional_correctness'))} | "
                    f"{reward_hack_str} | {row.get('accuracy', '—')} | {latency_str} | "
                    f"{row.get('failure_count', 0)} | `{row.get('artifact_dir', '')}` |"
                )
    else:
        lines.append("No `eval_status.json` files found.")
    lines.append("")

    return "\n".join(lines) + "\n"


def write_reports(run: str | Path, output_dir: str = "runs") -> tuple[Path, Path, dict[str, Any]]:
    """Write report.json and report.md for one run directory."""
    run_dir = _resolve_run_dir(run, output_dir=output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = build_report_payload(run_dir)
    report_json_path = run_dir / "report.json"
    report_md_path = run_dir / "report.md"

    report_json_path.write_text(json.dumps(payload, indent=2) + "\n")
    report_md_path.write_text(render_report_markdown(payload))

    return report_json_path, report_md_path, payload


def print_summary(payload: dict[str, Any]) -> None:
    """Print a concise summary for one aggregated report payload."""
    summary = payload["summary"]
    pass_at_1 = summary.get("pass_at_1")
    pass_at_5 = summary.get("pass_at_5")
    pass_at_1_text = f"{100 * pass_at_1:.1f}%" if pass_at_1 is not None else "n/a"
    pass_at_5_text = f"{100 * pass_at_5:.1f}%" if pass_at_5 is not None else "n/a"
    print(
        "Report:"
        f" evaluated={summary['evaluated_entries']}"
        f" | passed={summary['passed']}"
        f" | Pass@1={pass_at_1_text}"
        f" | Pass@5={pass_at_5_text}"
        f" | syntax_correctness={summary['syntax_correctness']}"
        f" | functional_correctness={summary['functional_correctness']}"
        f" | reward_hack={summary['reward_hack_detected']}"
    )
