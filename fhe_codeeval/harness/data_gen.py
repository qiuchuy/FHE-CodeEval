"""
Test data generation for FHE benchmark cases.

Cases with ``sampling_strategy: uniform`` use the full specified range for
every sample. Cases with ``sampling_strategy: stratified`` split that range
into equal-width sub-intervals and draw the ``i``-th sample from the ``i``-th
sub-interval. Seeds are derived from each case's stable ``seed_id`` so public
category or case-name changes do not alter the experimental inputs.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch


def _seed_from_case_id(case_id: str, offset: int = 0) -> int:
    """Derive a deterministic seed from a case id string."""
    digest = hashlib.sha256(case_id.encode()).hexdigest()
    return (int(digest[:8], 16) + offset) & 0xFFFFFFFF


def _should_stratify_input_range(case_meta: dict) -> bool:
    """Return whether this case explicitly requests stratified sampling."""
    strategy = case_meta["sampling_strategy"]
    if strategy not in {"uniform", "stratified"}:
        raise ValueError(f"Unsupported sampling_strategy {strategy!r} for {case_meta['id']}")
    return strategy == "stratified"


def _range_for_sample(
    spec: dict,
    *,
    case_meta: dict,
    sample_idx: int,
    n_samples: int,
) -> tuple[float, float]:
    """
    Return the numeric range to use for one generated sample.

    Stratified cases split the declared range into ``n_samples`` equal-width
    intervals and assign one interval per generated sample. Uniform cases use
    the full range for every sample.
    """
    lo, hi = spec.get("range", [-1, 1])
    if (
        not _should_stratify_input_range(case_meta)
        or n_samples <= 1
        or spec.get("dtype", "float32") == "int64"
        or hi <= lo
    ):
        return lo, hi

    width = (hi - lo) / n_samples
    sub_lo = lo + sample_idx * width
    sub_hi = hi if sample_idx == n_samples - 1 else sub_lo + width
    return sub_lo, sub_hi


def _make_tensor(
    spec: dict,
    seed: int,
    value_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Generate a single tensor according to an input spec."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    shape = spec["shape"]
    dtype_str = spec.get("dtype", "float32")
    lo, hi = value_range if value_range is not None else spec.get("range", [-1, 1])

    if dtype_str == "int64":
        # Uniform integer in [lo, hi] inclusive
        return torch.randint(int(lo), int(hi) + 1, shape, generator=rng)

    # float32
    t = torch.rand(shape, dtype=torch.float32, generator=rng)
    return t * (hi - lo) + lo


def generate_inputs(case_meta: dict) -> list[dict[str, Any]]:
    """
    Returns a list of generated input dicts for a spec-based case.

    Each dict maps argument name → tensor (or scalar for type: scalar args).

    Args:
        case_meta: A single case entry from benchmarks.yaml.
    Returns:
        List of length n_test_inputs, each element is {arg_name: value}.
    """
    n = case_meta.get("n_test_inputs", 5)
    seed_id = case_meta["seed_id"]
    results = []

    for i in range(n):
        sample: dict[str, Any] = {}
        for spec in case_meta["input_specs"]:
            if spec.get("name") == "model":
                # Model objects are constructed deterministically by the
                # evaluator and are not random input data.
                continue
            if spec.get("type") == "scalar":
                sample[spec["name"]] = spec["value"]
            else:
                # Each argument gets a distinct seed so args are independent
                arg_seed = _seed_from_case_id(f"{seed_id}:{spec['name']}", offset=i)
                value_range = _range_for_sample(
                    spec,
                    case_meta=case_meta,
                    sample_idx=i,
                    n_samples=n,
                )
                sample[spec["name"]] = _make_tensor(
                    spec,
                    arg_seed,
                    value_range=value_range,
                )
        results.append(sample)

    return results


def build_case_args(
    case_meta: dict,
    sample: dict[str, Any],
    model: Any | None = None,
) -> list[Any]:
    """Build ordered torch_kernel/fhe_kernel args from a generated sample."""
    args: list[Any] = []
    for spec in case_meta.get("input_specs", []):
        name = spec["name"]
        if name == "model":
            args.append(model)
        else:
            args.append(sample[name])
    return args
