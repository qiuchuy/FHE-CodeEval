"""
Activation range profiler for FHE benchmark network cases.

Runs the torch_kernel in cleartext with forward hooks to record per-layer
min/max of nonlinear activation *inputs* across random input samples.  These
ranges are the `[a, b]` domain for EvalChebyshevSeries in FHE kernels.

Results are cached to ~/.cache/fhe_codeeval/ranges/{case_id_safe}_ranges.json
so subsequent prompt builds are instant.

Usage:
    from harness.range_profiler import profile_activation_ranges
    ranges = profile_activation_ranges(case_meta, torch_kernel_fn, model, n_samples=100)
    # {"act": (-0.12, 3.52), "fc1.act": (-0.05, 2.11), ...}
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from fhe_codeeval.harness.data_gen import build_case_args

_NONLINEAR_TYPES = (
    nn.ReLU,
    nn.ReLU6,
    nn.GELU,
    nn.Tanh,
    nn.Sigmoid,
    nn.SiLU,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.Softmax,
    nn.LogSoftmax,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
)

_CACHE_DIR = Path.home() / ".cache" / "fhe_codeeval" / "ranges"


def _cache_path(case_meta: dict) -> Path:
    key = case_meta["id"].replace("/", "_")
    return _CACHE_DIR / f"{key}_ranges.json"


def _torch_ref_hash(case_meta: dict) -> str:
    if "torch_ref" not in case_meta:
        return ""
    path = Path(case_meta["torch_ref"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cache(case_meta: dict) -> Optional[dict[str, tuple[float, float]]]:
    p = _cache_path(case_meta)
    if p.exists():
        try:
            with open(p) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if "ranges" not in raw or raw.get("torch_ref_hash") != _torch_ref_hash(case_meta):
            return None
        return {k: tuple(v) for k, v in raw["ranges"].items()}
    return None


def _save_cache(case_meta: dict, ranges: dict[str, tuple[float, float]]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = _cache_path(case_meta)
    payload = {
        "torch_ref_hash": _torch_ref_hash(case_meta),
        "ranges": {k: list(v) for k, v in ranges.items()},
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=_CACHE_DIR,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def invalidate_cache(case_id: str) -> None:
    """Delete cached ranges for a case (call after changing model weights)."""
    key = case_id.replace("/", "_")
    p = _CACHE_DIR / f"{key}_ranges.json"
    if p.exists():
        p.unlink()


def profile_activation_ranges(
    case_meta: dict,
    torch_kernel_fn: Callable,
    model: Optional[Any],
    n_samples: int = 200,
    inputs: Optional[list[dict]] = None,
) -> dict[str, tuple[float, float]]:
    """
    Profile nonlinear activation input ranges in cleartext.

    Registers forward hooks on all ReLU / GELU / BatchNorm / LayerNorm / etc.
    submodules of ``model``, runs ``torch_kernel_fn`` on the supplied inputs,
    and records the global min/max seen at each layer's input.

    The ranges are the exact observed min/max (no margin added).

    Returns:
        Dict mapping ``"layer_name (TypeName)"`` → ``(lo, hi)`` after margin.
        Results are loaded from cache on subsequent calls.

    Args:
        case_meta:        Registry dict for the case.
        torch_kernel_fn:  The cleartext torch_kernel function.
        model:            The model instance (None for stateless operator cases).
        n_samples:        Max number of random samples generated when ``inputs`` is None.
        inputs:           Pre-generated input dicts (e.g. the exact evaluation
                          tensors from ``generate_inputs``).  When provided
                          these are used directly and ``n_samples`` is ignored.
    """
    cached = _load_cache(case_meta)
    if cached is not None:
        return cached

    if model is None:
        return {}

    hooks: list = []
    per_sample_calls: list[list[tuple[str, float, float]]] = []
    current_calls: list[tuple[str, float, float]] = []

    def _make_hook(name: str):
        def hook(module: nn.Module, inp, out):
            tensor = inp[0] if isinstance(inp, (tuple, list)) and len(inp) > 0 else None
            if tensor is None or not isinstance(tensor, torch.Tensor):
                return
            lo = float(tensor.detach().min().item())
            hi = float(tensor.detach().max().item())
            current_calls.append((name, lo, hi))

        return hook

    for name, module in model.named_modules():
        if isinstance(module, _NONLINEAR_TYPES) or getattr(module, "_fhe_profile_nonlinear", False):
            label = f"{name} ({type(module).__name__})" if name else type(module).__name__
            h = module.register_forward_hook(_make_hook(label))
            hooks.append(h)

    if not hooks:
        return {}

    from fhe_codeeval.harness.data_gen import generate_inputs

    def _run_one_sample(fn_call):
        """Execute one forward pass, tracking hook calls separately."""
        current_calls.clear()
        try:
            fn_call()
        except Exception:
            pass
        if current_calls:
            per_sample_calls.append(list(current_calls))

    def _run_with_inputs(input_dicts: list[dict]) -> None:
        """Forward-pass each pre-generated input dict through the kernel."""
        with torch.no_grad():
            for sample in input_dicts:
                args = build_case_args(case_meta, sample, model=model)
                _run_one_sample(lambda args=args: torch_kernel_fn(*args))

    if inputs is not None:
        _run_with_inputs(inputs)
    else:
        runtime_specs = [s for s in case_meta.get("input_specs", []) if s.get("name") != "model"]
        if not runtime_specs:
            for h in hooks:
                h.remove()
            return {}
        synth_meta = {**case_meta, "n_test_inputs": 1, "input_specs": runtime_specs}
        with torch.no_grad():
            for i in range(n_samples):
                gen = generate_inputs(synth_meta)
                if not gen:
                    break
                sample = gen[0]
                args = build_case_args(case_meta, sample, model=model)
                _run_one_sample(lambda args=args: torch_kernel_fn(*args))

    for h in hooks:
        h.remove()

    # Aggregate per-call-position ranges across all samples.
    # This correctly distinguishes shared modules called multiple times per forward pass.
    from collections import Counter

    pos_ranges: dict[int, dict] = {}
    for sample_calls in per_sample_calls:
        for pos, (name, lo, hi) in enumerate(sample_calls):
            if pos not in pos_ranges:
                pos_ranges[pos] = {"name": name, "lo": lo, "hi": hi}
            else:
                pos_ranges[pos]["lo"] = min(pos_ranges[pos]["lo"], lo)
                pos_ranges[pos]["hi"] = max(pos_ranges[pos]["hi"], hi)

    name_counts = Counter(r["name"] for r in pos_ranges.values())
    name_seen: dict[str, int] = {}

    result: dict[str, tuple[float, float]] = {}
    for pos in sorted(pos_ranges.keys()):
        r = pos_ranges[pos]
        name = r["name"]
        name_seen[name] = name_seen.get(name, 0) + 1
        if name_counts[name] > 1:
            label = f"{name} #{name_seen[name]}"
        else:
            label = name
        result[label] = (round(r["lo"], 6), round(r["hi"], 6))

    _save_cache(case_meta, result)
    return result


def format_ranges_table(ranges: dict[str, tuple[float, float]]) -> str:
    """
    Format activation ranges as a Markdown table for inclusion in prompts.

    Returns empty string if ranges is empty.
    """
    if not ranges:
        return ""

    lines = [
        "### Observed Activation Ranges (cleartext profiling)",
        "",
        "Use these as the `[a, b]` domain for `cc.EvalChebyshevSeries(ct, coeffs, a, b)`.",
        "Ranges are exact observed min/max across all test inputs.",
        "",
        "| Layer | Range |",
        "|-------|-------|",
    ]
    for layer, (lo, hi) in ranges.items():
        lines.append(f"| `{layer}` | [{lo:.3f}, {hi:.3f}] |")

    lines.append("")
    lines.append("All profiled nonlinear operations must be implemented with HE operations inside `fhe_kernel()`.")
    lines.append(
        "`decrypt()` may only decrypt, unpack, cast, and reshape; it must not apply "
        "activations or any other task computation."
    )
    lines.append("")
    return "\n".join(lines)
