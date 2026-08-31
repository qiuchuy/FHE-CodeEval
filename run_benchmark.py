#!/usr/bin/env python3
"""
FHE Benchmark — main CLI entry point.

Examples:
    # Copy the full template, edit it, then run by config
    mkdir -p configs/local
    cp configs/template.yaml configs/local/my_run.yaml
    uv run python run_benchmark.py --config configs/local/my_run.yaml

    # Evaluate one pre-existing fhe_kernel.py without creating batch reports
    uv run python evaluate.py --case packing-oriented-operators/matmul \\
        --fhe-kernel runs/run_agentic_001/matmul/fhe_kernel.py

Config YAML format:
    See configs/template.yaml for the full schema.

Output layout:
    runs/
      {run_id}/
        report.json
        report.md
        {case_name}/          # e.g. matmul, relu, linear-softmax
          fhe_kernel.py
          eval_status.json
          feedback_attempts.json  # feedback mode only
          prompt.txt / CLAUDE.md / ...
        {case_name}/{sample_id}/  # when sampling_num is set; sample_id = 1..n
          ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from fhe_codeeval.harness.evaluator import EvalResult
    from fhe_codeeval.llm.client import ChatMessage


_DEFAULT_REGISTRY = str(Path(__file__).resolve().parent / "fhe_codeeval" / "registry" / "benchmarks.yaml")


def _load_registry(registry_path: str = _DEFAULT_REGISTRY) -> list[dict]:
    with open(registry_path) as f:
        return yaml.safe_load(f)


def _case_short_name(case_id: str) -> str:
    """Return the leaf name of a case id (strip category prefix).

    "packing-oriented-operators/matmul" -> "matmul"
    "neural-network-workloads/mlp"      -> "mlp"
    """
    return case_id.split("/")[-1]


def _case_output_dir(run_id: str, case_id: str, sample_id: str | None = None) -> Path:
    """Return the artifact directory for one case/sample."""
    path = Path("runs") / run_id / _case_short_name(case_id)
    if sample_id is not None:
        path = path / str(sample_id)
    return path


def _write_case_prompt_txt(output_dir: Path, prompt: str) -> None:
    """Persist the initial LLM task prompt next to fhe_kernel.py (one_shot / feedback)."""
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")


def _write_cost_json(output_dir: Path, data: dict) -> None:
    (output_dir / "cost.json").write_text(json.dumps(data, indent=2) + "\n")


_MANIFEST_MODEL_FIELDS = (
    "provider",
    "model_id",
    "max_tokens",
    "temperature",
    "enable_thinking",
    "use_max_completion_tokens",
    "timeout_seconds",
    "retries",
)


def _manifest_model_settings(resolved_model: dict) -> dict:
    """Return reproducibility settings that cannot contain credentials.

    Endpoint values, API keys, and arbitrary provider parameters are omitted
    instead of redacted: URLs and nested request headers can themselves carry
    credentials, so a denylist is not sufficient here.
    """
    return {field: resolved_model[field] for field in _MANIFEST_MODEL_FIELDS if field in resolved_model}


def _git_revision() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _write_run_manifest(
    run_id: str,
    args: argparse.Namespace,
    selected: list[dict],
    llm_config: dict,
) -> Path:
    """Persist a credential-free snapshot of the resolved experiment setup."""
    from fhe_codeeval.llm.models import get_model_config

    revision, dirty = _git_revision()
    resolved_model = get_model_config(args.model, overrides=llm_config)
    payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_revision": revision,
        "git_dirty": dirty,
        "method": args.method,
        "model": args.model,
        "resolved_model": _manifest_model_settings(resolved_model),
        "model_api_key_env": args.model_api_key_env,
        "model_base_url_env": args.model_base_url_env,
        "cases": [case["id"] for case in selected],
        "sampling_num": args.sampling_num,
        "feedback_rounds": args.feedback_rounds,
        "num_workers": args.num_workers,
        "simulate": args.simulate,
        "no_eval": args.no_eval,
        "setup_only": args.setup_only,
    }
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _read_agentic_cost(trajectory_path: Path, case_id: str, model: str) -> dict:
    """Parse cost/usage from the result event in a raw_trajectory.jsonl file."""
    result_event: dict = {}
    try:
        with open(trajectory_path) as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("type") == "result":
                    result_event = obj
                    break
    except (OSError, json.JSONDecodeError):
        pass

    usage = result_event.get("usage", {})
    return {
        "method": "agentic",
        "model": model,
        "case_id": case_id,
        "e2e_runtime_s": round(result_event["duration_ms"] / 1000, 3) if "duration_ms" in result_event else None,
        "token_usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        },
    }


def _sampling_labels(sampling_num: int | None) -> list[str | None]:
    """Return [None] for the default single output, or ["1", ..., "n"]."""
    if sampling_num is None:
        return [None]
    return [str(i) for i in range(1, sampling_num + 1)]


def _none_if_blank(value):
    """Treat null-ish config strings as missing values."""
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def _should_skip(
    run_id: str,
    case_id: str,
    sample_id: str | None,
    feedback_rounds: int,
    method: str = "feedback",
) -> tuple[bool, str]:
    """Decide whether a case/sample can be skipped.

    Returns (should_skip, reason).  A case is skipped when:
      * eval_status.json exists AND the case passed, OR
      * For feedback method only: eval_status.json exists, the case failed,
        AND the feedback attempt count has reached the maximum.
      * For non-feedback methods (agentic, one_shot): eval_status.json
        exists (skip regardless of pass/fail).
    """
    output_dir = _case_output_dir(run_id, case_id, sample_id)
    eval_status_path = output_dir / "eval_status.json"
    if not eval_status_path.exists():
        return False, ""

    try:
        status = json.loads(eval_status_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False, ""

    if status.get("passed"):
        return True, "passed"

    if method != "feedback":
        return True, "already evaluated"

    max_attempts = feedback_rounds + 1
    attempts_path = output_dir / "feedback_attempts.json"
    if attempts_path.exists():
        try:
            attempts = json.loads(attempts_path.read_text())
            if isinstance(attempts, list) and len(attempts) >= max_attempts:
                return True, f"failed after {len(attempts)}/{max_attempts} attempts"
        except (json.JSONDecodeError, OSError):
            pass

    return False, ""


def _llm_config_from_args(args: argparse.Namespace) -> dict:
    """Collect optional LLM endpoint overrides from CLI/config values."""
    mapping = {
        "provider": getattr(args, "model_provider", None),
        "model_id": getattr(args, "model_id", None),
        "base_url": getattr(args, "model_base_url", None),
        "base_url_env": getattr(args, "model_base_url_env", None),
        "api_key": getattr(args, "model_api_key", None),
        "api_key_env": getattr(args, "model_api_key_env", None),
        "max_tokens": getattr(args, "model_max_tokens", None),
        "temperature": getattr(args, "model_temperature", None),
        "enable_thinking": getattr(args, "model_enable_thinking", None),
        "extra_params": getattr(args, "model_extra_params", None),
        "timeout_seconds": getattr(args, "model_timeout_seconds", None),
        "retries": getattr(args, "model_retries", None),
        "use_max_completion_tokens": getattr(args, "model_use_max_completion_tokens", None),
        "agentic_timeout_seconds": getattr(args, "agentic_timeout_seconds", None),
    }
    return {key: value for key, raw_value in mapping.items() if (value := _none_if_blank(raw_value)) is not None}


def _apply_llm_env(env: dict[str, str], llm_config: dict | None) -> None:
    """Expose configured API credentials/base URLs to subprocess based LLM tools."""
    if not llm_config:
        return

    provider = str(llm_config.get("provider", "")).lower().replace("_", "-")
    if provider in {"openai-compatible", "glm", "zhipu", "bigmodel"}:
        provider = "openai"
    elif provider in {"anthropic-compatible", "claude"}:
        provider = "anthropic"

    api_key = _none_if_blank(llm_config.get("api_key"))
    api_key_env = _none_if_blank(llm_config.get("api_key_env"))
    base_url = _none_if_blank(llm_config.get("base_url"))
    base_url_env = _none_if_blank(llm_config.get("base_url_env"))

    if not api_key and api_key_env:
        api_key = _none_if_blank(env.get(str(api_key_env)))
    if not base_url and base_url_env:
        base_url = _none_if_blank(env.get(str(base_url_env)))

    if api_key and api_key_env:
        env[str(api_key_env)] = str(api_key)
    if base_url and base_url_env:
        env[str(base_url_env)] = str(base_url)

    if provider == "anthropic":
        if api_key:
            env["ANTHROPIC_API_KEY"] = str(api_key)
        if base_url:
            env["ANTHROPIC_BASE_URL"] = str(base_url)
    elif provider == "openai":
        if api_key:
            env["OPENAI_API_KEY"] = str(api_key)
        if base_url:
            env["OPENAI_BASE_URL"] = str(base_url)


def _filter_cases(registry: list[dict], cases: list[str]) -> list[dict]:
    """Filter registry entries by case id and reject unknown ids."""
    if not cases:
        return list(registry)
    known = {case["id"] for case in registry}
    unknown = sorted(set(cases) - known)
    if unknown:
        raise ValueError(f"Unknown benchmark case id(s): {', '.join(unknown)}")
    return [c for c in registry if c["id"] in cases]


def _exclude_case_prefixes(selected: list[dict], prefixes: list[str] | None) -> list[dict]:
    """Drop registry entries whose id starts with any of the given prefixes."""
    if not prefixes:
        return selected
    return [c for c in selected if not any(str(c["id"]).startswith(p) for p in prefixes)]


def _write_run_report(run_id: str, *, no_eval: bool = False) -> None:
    """Generate report.json/report.md for one run directory from eval_status.json artifacts."""
    from fhe_codeeval.harness.reporter import collect_eval_statuses, print_summary, write_reports

    statuses = collect_eval_statuses(run_id)
    if statuses:
        report_json_path, report_md_path, report_payload = write_reports(run_id)
        print(f"\nReport JSON → {report_json_path}")
        print(f"Report MD   → {report_md_path}")
        print_summary(report_payload)
    elif no_eval:
        print("\nNo eval_status.json files found; report skipped for no-eval run.")


def _run_agentic(
    workspace: Path,
    case_id: str,
    model: str,
    llm_config: dict | None = None,
) -> bool:
    """
    Run Claude Code in headless mode inside the workspace.

    Saves the full Claude Code stream to ``workspace/raw_trajectory.jsonl``.

    Returns True if fhe_kernel.py was generated successfully.
    """
    workspace_abs = str(workspace.resolve())
    timeout_seconds = int((llm_config or {}).get("agentic_timeout_seconds", 5400))
    cmd = [
        "claude",
        "--dangerously-skip-permissions",
        "-p",
        "Read prompt.txt and follow the instructions in it.",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        "--tools",
        "Bash,Read,Write,Edit,Grep,Glob",
    ]

    env = os.environ.copy()
    env["IS_SANDBOX"] = "1"
    _apply_llm_env(env, llm_config)

    trajectory_path = workspace / "raw_trajectory.jsonl"

    print(f"  [agentic] Running Claude Code (headless) in {workspace} ...")
    print(f"  [agentic] Trajectory: {trajectory_path}")
    t0 = time.perf_counter()
    try:
        with open(trajectory_path, "w") as traj_f:
            proc = subprocess.Popen(
                cmd,
                cwd=workspace_abs,
                env=env,
                stdout=traj_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                proc.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                elapsed = time.perf_counter() - t0
                print(f"  [agentic] Claude Code exceeded {timeout_seconds}s and was stopped ({elapsed:.1f}s)")
                print(f"  [agentic] See trajectory: {trajectory_path}")
                return False

        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            print(f"  [agentic] Claude Code exited with code {proc.returncode} ({elapsed:.1f}s)")
            print(f"  [agentic] See trajectory: {trajectory_path}")
            return False
        print(f"  [agentic] Claude Code finished in {elapsed:.1f}s")
    except FileNotFoundError:
        print(
            "  [agentic] ERROR: 'claude' CLI not found. "
            "Install Claude Code (https://code.claude.com) or use --setup-only."
        )
        return False

    fhe_kernel_path = workspace / "fhe_kernel.py"
    if fhe_kernel_path.exists() and fhe_kernel_path.stat().st_size > 100:
        return True

    print("  [agentic] WARNING: fhe_kernel.py was not generated or is too small")
    return False


def _feedback_messages(
    prompt: str,
    history: list[dict],
) -> list["ChatMessage"]:
    """Build feedback chat history as user(prompt), assistant(code), user(feedback)."""
    messages: list[ChatMessage] = [{"role": "user", "content": prompt}]
    for turn in history:
        messages.append({"role": "assistant", "content": turn["assistant_code"]})
        messages.append({"role": "user", "content": turn["feedback_to_model"]})
    return messages


def _build_feedback_message(result: "EvalResult", payload: dict) -> str:
    """Convert an EvalResult into the next user message for feedback mode.

    Reports execution errors only (import, runtime, accuracy).
    Reward-hack detection details are never exposed to the model.
    """
    status: dict = {
        "passed": payload["passed"],
        "syntax_correctness": payload["syntax_correctness"],
        "functional_correctness": payload["functional_correctness"],
        "accuracy": payload["accuracy"],
    }
    if payload.get("latency_ms") is not None:
        status["latency_ms"] = payload["latency_ms"]

    lines = [
        "Your previous `fhe_kernel.py` did not pass the evaluator.",
        "",
        "Evaluation status:",
        json.dumps(status, indent=2, default=str),
    ]

    public_failures = [
        failure
        for failure in payload.get("failures", [])
        if failure.get("stage") not in {"static_check", "execution_check"}
    ]

    if not result.compiled and result.compile_error:
        lines.extend(["", "Import/compile error:", result.compile_error])
    elif public_failures:
        lines.extend(["", "Failures:"])
        for f in public_failures:
            if "error" in f:
                idx = f.get("input_idx", f.get("stage", "?"))
                lines.append(f"- [{idx}] {f['error']}")
            elif "max_abs_diff" in f:
                lines.append(f"- [input {f['input_idx']}] max_abs_diff={f['max_abs_diff']}")

    lines.extend(
        [
            "",
            "Please fix the issue and respond with the FULL corrected `fhe_kernel.py`.",
            "Return ONLY valid Python code, no markdown fences and no explanation.",
        ]
    )
    return "\n".join(lines)


def _write_feedback_attempts(attempts_path: Path, attempts: list[dict]) -> None:
    attempts_path.write_text(json.dumps(attempts, indent=2, default=str) + "\n")


def _annotate_sample(result: "EvalResult", sample_id: str | None) -> None:
    """Attach sample metadata to EvalResult objects when sampling is active."""
    if sample_id is not None:
        result.sample_id = sample_id


def _run_feedback_and_evaluate(
    case_id: str,
    model: str,
    run_id: str,
    dry_run: bool,
    no_eval: bool,
    run_real: bool,
    feedback_rounds: int,
    sample_id: str | None = None,
    llm_config: dict | None = None,
) -> "EvalResult | None":
    from fhe_codeeval.harness.evaluator import build_payload, evaluate, write_eval_status
    from fhe_codeeval.llm.client import generate_fhe_kernel_from_messages
    from fhe_codeeval.prompts.generator import build_prompt

    # Network/compositional prompts may profile activations on first use;
    # flush so this is visible even when tqdm owns stderr.
    print(
        f"  [prompt] Building prompt for {case_id} "
        f"(first network run may profile activations; cached under ~/.cache/fhe_codeeval/ranges/) ...",
        flush=True,
    )
    prompt = str(build_prompt(case_id, method="feedback", run_id=run_id))
    print(f"  [prompt] Done ({len(prompt)} chars).", flush=True)
    if dry_run:
        print(f"\n{'=' * 60}")
        sample_label = f"  |  SAMPLE: {sample_id}" if sample_id is not None else ""
        print(f"CASE: {case_id}  |  METHOD: feedback  |  MODEL: {model}{sample_label}")
        print("─" * 60)
        print(prompt[:2000])
        if len(prompt) > 2000:
            print(f"... [{len(prompt) - 2000} chars truncated]")
        print(f"[feedback] Max correction rounds: {feedback_rounds}")
        return None

    output_dir = _case_output_dir(run_id, case_id, sample_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_case_prompt_txt(output_dir, prompt)
    fhe_kernel_path = output_dir / "fhe_kernel.py"
    attempts_path = output_dir / "feedback_attempts.json"

    from fhe_codeeval.llm.client import _empty_usage, _merge_usage

    history: list[dict] = []
    attempts: list[dict] = []
    last_result: EvalResult | None = None
    max_attempts = feedback_rounds + 1
    cumulative_usage = _empty_usage()
    e2e_t0 = time.perf_counter()

    for attempt in range(1, max_attempts + 1):
        messages = _feedback_messages(prompt, history)
        label = "initial" if attempt == 1 else f"feedback {attempt - 1}/{feedback_rounds}"
        print(f"  [feedback] Calling {model} ({label}) ...", flush=True)
        t0 = time.perf_counter()
        code, usage = generate_fhe_kernel_from_messages(
            messages,
            model,
            output_path=str(fhe_kernel_path),
            config_overrides=llm_config,
        )
        elapsed = time.perf_counter() - t0
        cumulative_usage = _merge_usage(cumulative_usage, usage)
        print(f"  [feedback] Attempt {attempt}: {len(code)} chars in {elapsed:.1f}s")

        record = {
            "attempt": attempt,
            "request_messages": messages,
            "code": code,
            "evaluation": None,
            "feedback_sent_to_model": None,
        }

        if no_eval:
            attempts.append(record)
            _write_feedback_attempts(attempts_path, attempts)
            _write_cost_json(
                output_dir,
                {
                    "method": "feedback",
                    "model": model,
                    "case_id": case_id,
                    "e2e_runtime_s": round(time.perf_counter() - e2e_t0, 3),
                    "token_usage": cumulative_usage,
                },
            )
            print("  [feedback] no_eval=true; skipping evaluator feedback")
            return None

        result = evaluate(
            case_id,
            str(fhe_kernel_path),
            method="feedback",
            model=model,
            run_real=run_real,
        )
        _annotate_sample(result, sample_id)
        payload = build_payload(result)
        write_eval_status(payload, str(fhe_kernel_path))
        last_result = result
        record["evaluation"] = payload

        status = "✓" if payload["passed"] else "✗"
        print(
            f"  {status} attempt={attempt}/{max_attempts}  compiled={result.compiled}  "
            f"accuracy={payload['accuracy']}  latency={result.latency_ms}ms  "
            f"reward_hack={result.reward_hack_detected}"
        )

        # Hacking-detection details are not part of iterative feedback. Once
        # the terminal candidate is functionally correct, retain its hacking
        # result for reporting and stop the repair trajectory.
        if payload["functional_correctness"]:
            attempts.append(record)
            _write_feedback_attempts(attempts_path, attempts)
            _write_cost_json(
                output_dir,
                {
                    "method": "feedback",
                    "model": model,
                    "case_id": case_id,
                    "e2e_runtime_s": round(time.perf_counter() - e2e_t0, 3),
                    "token_usage": cumulative_usage,
                },
            )
            return result

        if attempt < max_attempts:
            feedback_to_model = _build_feedback_message(result, payload)
            history.append(
                {
                    "attempt": attempt,
                    "assistant_code": code,
                    "feedback_to_model": feedback_to_model,
                }
            )
            record["feedback_sent_to_model"] = feedback_to_model

        attempts.append(record)
        _write_feedback_attempts(attempts_path, attempts)

    _write_cost_json(
        output_dir,
        {
            "method": "feedback",
            "model": model,
            "case_id": case_id,
            "e2e_runtime_s": round(time.perf_counter() - e2e_t0, 3),
            "token_usage": cumulative_usage,
        },
    )
    return last_result


def _run_llm_and_evaluate(
    case: dict,
    method: str,
    model: str,
    run_id: str,
    dry_run: bool,
    no_eval: bool,
    run_real: bool,
    setup_only: bool = False,
    feedback_rounds: int = 2,
    sample_id: str | None = None,
    llm_config: dict | None = None,
) -> "EvalResult | None":
    from fhe_codeeval.harness.evaluator import build_payload, evaluate, write_eval_status
    from fhe_codeeval.prompts.generator import build_prompt

    case_id = case["id"]

    if method == "agentic":
        workspace = build_prompt(
            case_id,
            method="agentic",
            run_id=run_id,
            sample_id=sample_id,
            simulate=not run_real,
        )
        print(f"  [agentic] Workspace created: {workspace}")

        if setup_only or dry_run:
            print("  [agentic] Setup only — run manually or re-run without --setup-only")
            return None

        agent_model = (llm_config or {}).get("model_id") or model
        success = _run_agentic(workspace, case_id, str(agent_model), llm_config=llm_config)

        trajectory_path = workspace / "raw_trajectory.jsonl"
        cost_data = _read_agentic_cost(trajectory_path, case_id, str(agent_model))
        _write_cost_json(workspace, cost_data)

        if not success:
            raise RuntimeError(f"Agentic generation failed for {case_id}")
        if no_eval:
            return None

        fhe_kernel_path = workspace / "fhe_kernel.py"
        result = evaluate(
            case_id,
            str(fhe_kernel_path),
            method=method,
            model=model,
            run_real=run_real,
        )
        _annotate_sample(result, sample_id)
        payload = build_payload(result)
        write_eval_status(payload, str(fhe_kernel_path))
        status = "✓" if payload["passed"] else "✗"
        print(
            f"  {status} compiled={result.compiled}  accuracy={payload['accuracy']}  "
            f"latency={result.latency_ms}ms  "
            f"reward_hack={result.reward_hack_detected}"
        )
        return result

    if method == "feedback":
        return _run_feedback_and_evaluate(
            case_id,
            model,
            run_id,
            dry_run,
            no_eval,
            run_real,
            feedback_rounds,
            sample_id=sample_id,
            llm_config=llm_config,
        )

    # One-shot uses the same canonical prompt/message structure as feedback.
    from fhe_codeeval.prompts.generator import build_prompt_messages

    messages = build_prompt_messages(case_id, method=method)
    prompt = messages[-1]["content"]

    if dry_run:
        print(f"\n{'=' * 60}")
        sample_label = f"  |  SAMPLE: {sample_id}" if sample_id is not None else ""
        print(f"CASE: {case_id}  |  METHOD: {method}  |  MODEL: {model}{sample_label}")
        print("─" * 60)
        print(prompt[:2000])
        if len(prompt) > 2000:
            print(f"... [{len(prompt) - 2000} chars truncated]")
        return None

    # Output dir: runs/{run_id}/{case_name}/ or runs/{run_id}/{case_name}/{sample_id}/
    output_dir = _case_output_dir(run_id, case_id, sample_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_case_prompt_txt(output_dir, str(prompt))
    fhe_kernel_path = output_dir / "fhe_kernel.py"

    print(f"  Calling {model} ...", flush=True)
    t0 = time.perf_counter()
    from fhe_codeeval.llm.client import generate_fhe_kernel_from_messages

    code, usage = generate_fhe_kernel_from_messages(
        messages,
        model,
        output_path=str(fhe_kernel_path),
        config_overrides=llm_config,
    )
    elapsed = time.perf_counter() - t0
    print(f"  LLM response: {len(code)} chars in {elapsed:.1f}s → {fhe_kernel_path}", flush=True)

    if no_eval:
        _write_cost_json(
            output_dir,
            {
                "method": method,
                "model": model,
                "case_id": case_id,
                "e2e_runtime_s": round(elapsed, 3),
                "token_usage": usage,
            },
        )
        return None

    eval_t0 = time.perf_counter()
    result = evaluate(
        case_id,
        str(fhe_kernel_path),
        method=method,
        model=model,
        run_real=run_real,
    )
    e2e_runtime_s = round(elapsed + (time.perf_counter() - eval_t0), 3)
    _annotate_sample(result, sample_id)
    payload = build_payload(result)
    write_eval_status(payload, str(fhe_kernel_path))
    _write_cost_json(
        output_dir,
        {
            "method": method,
            "model": model,
            "case_id": case_id,
            "e2e_runtime_s": e2e_runtime_s,
            "token_usage": usage,
        },
    )
    status = "✓" if payload["passed"] else "✗"
    print(
        f"  {status} compiled={result.compiled}  accuracy={payload['accuracy']}  "
        f"latency={result.latency_ms}ms  "
        f"reward_hack={result.reward_hack_detected}"
    )
    return result


def _load_config(config_path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def _run_one_case(
    case: dict,
    method: str,
    model: str,
    run_id: str,
    dry_run: bool,
    no_eval: bool,
    setup_only: bool,
    feedback_rounds: int,
    sample_id: str | None,
    llm_config: dict | None,
    skip_completed: bool,
    simulate: bool,
) -> dict | None:
    """Top-level worker for ProcessPoolExecutor.

    Must be defined at module level (not a closure) so it is picklable.
    Returns a serialisable result dict or None.
    """
    case_id = case["id"]
    if skip_completed:
        skip, _reason = _should_skip(run_id, case_id, sample_id, feedback_rounds, method)
        if skip:
            return None

    result = _run_llm_and_evaluate(
        case,
        method,
        model,
        run_id,
        dry_run,
        no_eval,
        not simulate,
        setup_only=setup_only,
        feedback_rounds=feedback_rounds,
        sample_id=sample_id,
        llm_config=llm_config,
    )
    if result is None:
        return None
    from fhe_codeeval.harness.evaluator import build_payload

    return build_payload(result)


def _apply_config(args: argparse.Namespace, cfg: dict) -> None:
    """Apply config values as defaults — CLI flags always take priority."""
    _FIELDS = {
        "method": str,
        "model": str,
        "model_provider": str,
        "model_id": str,
        "model_base_url": str,
        "model_base_url_env": str,
        "model_api_key": str,
        "model_api_key_env": str,
        "model_max_tokens": int,
        "model_temperature": float,
        "model_enable_thinking": bool,
        "model_extra_params": dict,
        "model_timeout_seconds": float,
        "model_retries": int,
        "model_use_max_completion_tokens": bool,
        "agentic_timeout_seconds": int,
        "run_id": str,
        "cases": list,
        "exclude_prefixes": list,
        "simulate": bool,
        "dry_run": bool,
        "no_eval": bool,
        "setup_only": bool,
        "skip_completed": bool,
        "feedback_rounds": int,
        "sampling_num": int,
        "num_workers": int,
    }
    for key, typ in _FIELDS.items():
        cfg_key = key.replace("_", "-") if key.replace("_", "-") in cfg else key
        cfg_val = cfg.get(cfg_key, cfg.get(key))
        if cfg_val is None:
            continue
        cli_val = getattr(args, key, None)
        # CLI flag was explicitly set — keep it
        if typ is bool:
            if cli_val is None:
                setattr(args, key, bool(cfg_val))
        elif typ is list:
            if not cli_val:
                setattr(args, key, cfg_val if isinstance(cfg_val, list) else [cfg_val])
        else:
            if cli_val is None:
                setattr(args, key, cfg_val)


def main():
    # ── Main argument parser ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="FHE Benchmark — evaluate LLM ability to write OpenFHE CKKS code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. CLI flags override config values.",
    )

    # Case selection
    parser.add_argument(
        "--cases",
        type=str,
        nargs="+",
        help="Specific case ids (e.g. packing-oriented-operators/matmul "
        "approximation-oriented-operators/relu). "
        "Omit to run all cases.",
    )
    parser.add_argument(
        "--exclude-prefix",
        type=str,
        action="append",
        default=None,
        dest="exclude_prefixes",
        help="Skip case ids starting with this prefix (repeatable)",
    )

    # Prompt method
    parser.add_argument(
        "--method",
        type=str,
        choices=["one_shot", "feedback", "agentic"],
        default=None,
        help="Prompt method (default: one_shot)",
    )

    # LLM config
    parser.add_argument("--model", type=str, default=None, help="LLM model name (see fhe_codeeval/llm/models.py)")
    parser.add_argument(
        "--model-provider",
        type=str,
        default=None,
        help="LLM provider override: anthropic, openai, or compatible aliases",
    )
    parser.add_argument("--model-id", type=str, default=None, help="Provider API model id. Defaults to --model")
    parser.add_argument("--model-base-url", type=str, default=None, help="Provider base URL override")
    parser.add_argument(
        "--model-base-url-env",
        type=str,
        default=None,
        help="Environment variable containing the provider base URL",
    )
    parser.add_argument(
        "--model-api-key",
        type=str,
        default=None,
        help="Provider API key override. Prefer config/env for real runs",
    )
    parser.add_argument(
        "--model-api-key-env",
        type=str,
        default=None,
        help="Environment variable containing the provider API key",
    )
    parser.add_argument(
        "--model-max-tokens",
        type=int,
        default=None,
        help="Maximum response tokens for API-backed modes",
    )
    parser.add_argument(
        "--model-temperature",
        type=float,
        default=None,
        help="Sampling temperature for API-backed modes",
    )
    parser.add_argument(
        "--model-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(openai-compatible) Request provider-side thinking/reasoning if supported",
    )
    parser.add_argument(
        "--model-use-max-completion-tokens",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Send the output cap as max_completion_tokens instead of max_tokens",
    )
    parser.add_argument(
        "--model-extra-params",
        type=json.loads,
        default=None,
        help="JSON object with provider-specific extra params, e.g. '{\"seed\": 42}'",
    )
    parser.add_argument(
        "--model-timeout-seconds",
        type=float,
        default=None,
        help="HTTP request timeout for API-backed LLM calls (default: 1800)",
    )
    parser.add_argument(
        "--model-retries",
        type=int,
        default=None,
        help="Retry attempts around each API-backed LLM call (default: 2)",
    )
    parser.add_argument(
        "--agentic-timeout-seconds",
        type=int,
        default=None,
        help="Claude Code wall-clock budget per agentic trial (default: 5400)",
    )

    # Run control
    parser.add_argument("--run-id", type=str, default=None, help="Unique run identifier (default: auto timestamp)")
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print prompts without making LLM API calls",
    )
    evaluation_group = parser.add_mutually_exclusive_group()
    evaluation_group.add_argument(
        "--no-eval",
        dest="no_eval",
        action="store_true",
        default=None,
        help="Generate fhe_kernel.py files but skip evaluation",
    )
    evaluation_group.add_argument(
        "--eval",
        dest="no_eval",
        action="store_false",
        default=None,
        help="Evaluate generated fhe_kernel.py files",
    )
    parser.add_argument(
        "--setup-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(agentic) Only create workspaces, don't run Claude Code",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip cases that passed or exhausted all feedback attempts",
    )
    parser.add_argument(
        "--feedback-rounds",
        type=int,
        default=None,
        help="(feedback) Number of evaluator-feedback correction rounds after the initial attempt",
    )
    parser.add_argument(
        "--sampling-num",
        type=int,
        default=None,
        help="Independent generations per benchmark case. When set to n, artifacts are written under case/1 ... case/n",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: 1, sequential)",
    )

    # Evaluation backend
    parser.add_argument(
        "--simulate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the plaintext simulator for correctness/latency instead of real OpenFHE",
    )
    args = parser.parse_args()

    # ── Apply config file defaults ────────────────────────────────────────────
    if args.config:
        cfg = _load_config(args.config)
        _apply_config(args, cfg)

    # Fill in final defaults for anything still None
    if args.method is None:
        args.method = "one_shot"
    if args.model is None:
        args.model = "claude-sonnet-4-6"
    if args.feedback_rounds is None:
        args.feedback_rounds = 5
    if args.feedback_rounds < 0:
        parser.error("--feedback-rounds must be >= 0")
    if isinstance(args.sampling_num, str) and args.sampling_num.lower() in {"none", "null", ""}:
        args.sampling_num = None
    if args.sampling_num is not None:
        try:
            args.sampling_num = int(args.sampling_num)
        except (TypeError, ValueError):
            parser.error("--sampling-num must be an integer or null")
        if args.sampling_num < 1:
            parser.error("--sampling-num must be >= 1")
    if args.num_workers is None:
        args.num_workers = 1
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.agentic_timeout_seconds is not None and args.agentic_timeout_seconds < 1:
        parser.error("--agentic-timeout-seconds must be >= 1")
    for boolean_field in ("dry_run", "no_eval", "setup_only", "skip_completed", "simulate"):
        if getattr(args, boolean_field) is None:
            setattr(args, boolean_field, False)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    llm_config = _llm_config_from_args(args)

    registry = _load_registry()

    # ── Main benchmark loop ───────────────────────────────────────────────────
    cases_filter = args.cases if args.cases else []
    exclude_prefixes = getattr(args, "exclude_prefixes", None) or []
    selected = _exclude_case_prefixes(_filter_cases(registry, cases_filter), exclude_prefixes)

    if not selected:
        print("No cases matched the given filters.", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run:
        manifest_path = _write_run_manifest(run_id, args, selected, llm_config)
        print(f"Run config → {manifest_path}")

    print(f"Run ID  : {run_id}")
    if args.config:
        print(f"Config  : {args.config}")
    print(f"Method  : {args.method}  |  Model: {args.model}  |  Cases: {len(selected)}")
    if args.sampling_num is not None:
        print(f"Samples : {args.sampling_num} per case")
    backend_label = (
        "static + simulator hacking checks; simulator correctness/latency"
        if args.simulate
        else "static + simulator hacking checks; real OpenFHE correctness/latency"
    )
    print(f"Backend : {backend_label}")
    print(f"Output  : runs/{run_id}/")
    if args.num_workers > 1:
        print(f"Workers : {args.num_workers}")
    print(f"Dry run : {args.dry_run}\n")

    all_results = []
    skipped = 0
    execution_errors = 0
    sample_ids = _sampling_labels(args.sampling_num)
    total_runs = len(selected) * len(sample_ids)
    progress_unit = "sample" if args.sampling_num is not None else "case"
    progress = tqdm(
        total=total_runs,
        desc="Benchmark",
        unit=progress_unit,
        dynamic_ncols=True,
        disable=args.dry_run,
    )

    if args.num_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        futures = {}
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for case in selected:
                for sample_id in sample_ids:
                    label = case["id"]
                    if sample_id is not None:
                        label += f"/{sample_id}"

                    if args.skip_completed:
                        skip, reason = _should_skip(
                            run_id,
                            case["id"],
                            sample_id,
                            args.feedback_rounds,
                            args.method,
                        )
                        if skip:
                            skipped += 1
                            progress.write(f"[skip] {label} ({reason})")
                            progress.set_postfix(skipped=skipped, refresh=False)
                            progress.update(1)
                            continue

                    fut = pool.submit(
                        _run_one_case,
                        case=case,
                        method=args.method,
                        model=args.model,
                        run_id=run_id,
                        dry_run=args.dry_run,
                        no_eval=args.no_eval,
                        setup_only=args.setup_only,
                        feedback_rounds=args.feedback_rounds,
                        sample_id=sample_id,
                        llm_config=llm_config,
                        skip_completed=args.skip_completed,
                        simulate=args.simulate,
                    )
                    futures[fut] = label

            try:
                for fut in as_completed(futures):
                    label = futures[fut]
                    try:
                        payload = fut.result()
                    except Exception as exc:
                        progress.write(f"[error] {label}: {exc}")
                        execution_errors += 1
                        progress.update(1)
                        continue
                    if payload is None:
                        skipped += 1
                        progress.write(f"[skip] {label}")
                        progress.set_postfix(skipped=skipped, refresh=False)
                    else:
                        all_results.append(payload)
                        status = "pass" if payload["passed"] else "fail"
                        progress.write(f"[{status}] {label}")
                        progress.set_postfix(results=len(all_results), skipped=skipped, refresh=False)
                    progress.update(1)
            finally:
                progress.close()
    else:
        run_index = 0
        try:
            for case in selected:
                for sample_id in sample_ids:
                    run_index += 1
                    sample_label = f"  sample {sample_id}/{args.sampling_num}" if sample_id is not None else ""
                    try:
                        if args.skip_completed:
                            skip, reason = _should_skip(
                                run_id, case["id"], sample_id, args.feedback_rounds, args.method
                            )
                            if skip:
                                line = f"[{run_index}/{total_runs}] {case['id']}{sample_label}  — skipped ({reason})"
                                if args.dry_run:
                                    print(line)
                                else:
                                    progress.write(line)
                                skipped += 1
                                progress.set_postfix(skipped=skipped, refresh=False)
                                continue

                        line = f"[{run_index}/{total_runs}] {case['id']}{sample_label}"
                        if args.dry_run:
                            print(line)
                        else:
                            progress.write(line)
                        result = _run_llm_and_evaluate(
                            case,
                            args.method,
                            args.model,
                            run_id,
                            args.dry_run,
                            args.no_eval,
                            not args.simulate,
                            setup_only=args.setup_only,
                            feedback_rounds=args.feedback_rounds,
                            sample_id=sample_id,
                            llm_config=llm_config,
                        )
                        if result is not None:
                            all_results.append(result)
                            progress.set_postfix(results=len(all_results), skipped=skipped, refresh=False)
                    finally:
                        progress.update(1)
        finally:
            progress.close()

    if skipped:
        skipped_unit = "samples" if args.sampling_num is not None else "cases"
        print(f"\nSkipped {skipped}/{total_runs} {skipped_unit} (passed or exhausted retries)")

    if not args.dry_run:
        _write_run_report(run_id, no_eval=args.no_eval)
    if execution_errors:
        raise SystemExit(f"{execution_errors} worker(s) failed before producing a result")


if __name__ == "__main__":
    main()
