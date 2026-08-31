"""Evaluation harness for the public FHE-CodeEval benchmark.

Every submission must implement the four module-level functions documented in
the paper.  Evaluation has three responsibilities:

1. run the static hacking check on the submitted source;
2. run the execution-based hacking check with the plaintext CKKS simulator;
3. measure correctness and ``fhe_kernel`` latency with real OpenFHE (or with
   the simulator when the caller explicitly sets ``run_real=False``).

The real OpenFHE stage performs only correctness and latency measurement.
Hacking decisions are made by the static and simulator-backed checks above.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

from fhe_codeeval.harness.data_gen import build_case_args, generate_inputs
from fhe_codeeval.harness.reward_hack import (
    REQUIRED_INTERFACE,
    RewardHackTag,
    check_execution,
    check_static,
)


@dataclass
class EvalResult:
    """Structured result for one generated program and one benchmark case."""

    case_id: str
    method: str
    model: str
    compiled: bool
    compile_error: str | None = None
    reward_hack_static: bool = False
    reward_hack_static_reason: str = ""
    reward_hack_execution: bool = False
    reward_hack_execution_reason: str = ""
    reward_hack_tags: list[str | RewardHackTag] = field(default_factory=list)
    accuracy: float | None = None
    accuracy_detail: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    validation_stages: dict[str, Any] = field(default_factory=dict)
    sample_id: str | None = None

    @property
    def reward_hack_detected(self) -> bool:
        return self.reward_hack_static or self.reward_hack_execution or bool(self.reward_hack_tags)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reward_hack_tags"] = _normalise_tags(self.reward_hack_tags)
        data["reward_hack_detected"] = self.reward_hack_detected
        if data.get("sample_id") is None:
            data.pop("sample_id", None)
        return data


@dataclass(frozen=True)
class KernelBundle:
    """The only generated-program interface supported by FHE-CodeEval."""

    make_context_fn: Callable[..., Any]
    encrypt_fn: Callable[..., Any]
    fhe_kernel_fn: Callable[..., Any]
    decrypt_fn: Callable[..., Any]


_PKG_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _PKG_DIR.parent
_DEFAULT_REGISTRY = str(_PKG_DIR / "registry" / "benchmarks.yaml")


def _normalise_tags(tags: list[str | RewardHackTag]) -> list[str]:
    values = [tag.value if isinstance(tag, RewardHackTag) else str(tag) for tag in tags]
    return list(dict.fromkeys(values))


def _clear_openfhe_modules() -> None:
    for name in list(sys.modules):
        if name == "openfhe" or name.startswith("openfhe."):
            sys.modules.pop(name, None)


@contextmanager
def _sanitised_candidate_environment():
    """Hide credential-like environment variables while generated code runs."""
    markers = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    removed = {key: value for key, value in os.environ.items() if any(marker in key.upper() for marker in markers)}
    for key in removed:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        os.environ.update(removed)


@contextmanager
def _openfhe_backend(*, simulate: bool):
    """Make ``import openfhe`` resolve to the selected backend temporarily.

    The simulator is published as ``simulate_ckks``.  Installing it under the
    ``openfhe`` module key here makes generated files use the same import on
    both backends and avoids relying on a patched OpenFHE wheel.
    """
    previous_modules = {
        name: module for name, module in sys.modules.items() if name == "openfhe" or name.startswith("openfhe.")
    }
    _clear_openfhe_modules()

    try:
        if simulate:
            simulator = importlib.import_module("simulate_ckks")
            sys.modules["openfhe"] = simulator
        yield
    finally:
        _clear_openfhe_modules()
        sys.modules.update(previous_modules)


def _load_registry(registry_path: str = _DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    with open(registry_path, encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    if not isinstance(registry, list):
        raise ValueError(f"Registry must contain a list of cases: {registry_path}")
    return registry


def _find_case(registry: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    for case in registry:
        if case["id"] == case_id:
            return case
    raise KeyError(f"Case {case_id!r} not found in registry")


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else _PROJECT_ROOT / resolved


def _load_ref_module(torch_ref_path: str):
    path = _resolve_project_path(torch_ref_path)
    spec = importlib.util.spec_from_file_location("_fhe_codeeval_torch_ref", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for reference: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fhe_module(
    fhe_kernel_path: str,
) -> tuple[KernelBundle | None, str | None]:
    """Import a generated module and require all four public entry points."""
    path = Path(fhe_kernel_path).resolve()
    module_name = "_fhe_codeeval_submission"
    sys.modules.pop(module_name, None)

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create import spec for submission: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        functions: dict[str, Callable[..., Any]] = {}
        missing: list[str] = []
        for name in REQUIRED_INTERFACE:
            value = getattr(module, name, None)
            if not callable(value):
                missing.append(name)
            else:
                functions[name] = value
        if missing:
            raise TypeError(
                "Submission must define callable module-level functions: "
                + ", ".join(REQUIRED_INTERFACE)
                + f". Missing/non-callable: {', '.join(missing)}."
            )

        return (
            KernelBundle(
                make_context_fn=functions["make_context"],
                encrypt_fn=functions["encrypt"],
                fhe_kernel_fn=functions["fhe_kernel"],
                decrypt_fn=functions["decrypt"],
            ),
            None,
        )
    except Exception:
        return None, traceback.format_exc()
    finally:
        sys.modules.pop(module_name, None)


def _create_model_for_case(ref_module: Any) -> Any | None:
    if not hasattr(ref_module, "create_model"):
        return None
    torch.manual_seed(42)
    return ref_module.create_model()


def _has_model(case_meta: dict[str, Any]) -> bool:
    return any(spec.get("name") == "model" for spec in case_meta["input_specs"])


def _generate_runtime_inputs(case_meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate inputs while excluding the separately constructed model object."""
    runtime_specs = [spec for spec in case_meta["input_specs"] if spec.get("name") != "model"]
    return generate_inputs({**case_meta, "input_specs": runtime_specs})


def _run_pipeline(
    bundle: KernelBundle,
    args: list[Any],
) -> tuple[Any, float]:
    """Execute one sample and time only the server-side homomorphic kernel."""
    cc, keys = bundle.make_context_fn()
    enc_inputs = bundle.encrypt_fn(cc, keys, *args)
    start = time.perf_counter()
    ct_out = bundle.fhe_kernel_fn(cc, keys, enc_inputs)
    latency_ms = (time.perf_counter() - start) * 1000.0
    output = bundle.decrypt_fn(cc, keys, ct_out)
    return output, latency_ms


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _run_accuracy(
    case_meta: dict[str, Any],
    torch_kernel_fn: Callable[..., Any],
    bundle: KernelBundle,
    model: Any | None,
) -> tuple[float, dict[str, Any], float | None]:
    """Compare all generated samples using the case's declared tolerance."""
    inputs = _generate_runtime_inputs(case_meta)
    atol = case_meta["accuracy"]["atol"]
    rtol = case_meta["accuracy"]["rtol"]
    passed = 0
    failures: list[dict[str, Any]] = []
    latencies: list[float] = []

    for index, sample in enumerate(inputs):
        args = build_case_args(case_meta, sample, model=model)
        try:
            reference = torch_kernel_fn(*args)
            generated, latency_ms = _run_pipeline(bundle, args)
            latencies.append(latency_ms)
            reference_tensor = _as_tensor(reference)
            generated_tensor = _as_tensor(generated)
            if reference_tensor.shape != generated_tensor.shape:
                failures.append(
                    {
                        "input_idx": index,
                        "expected_shape": list(reference_tensor.shape),
                        "actual_shape": list(generated_tensor.shape),
                    }
                )
            elif torch.allclose(
                reference_tensor,
                generated_tensor,
                atol=atol,
                rtol=rtol,
            ):
                passed += 1
            else:
                failures.append(
                    {
                        "input_idx": index,
                        "max_abs_diff": float(torch.max(torch.abs(reference_tensor - generated_tensor))),
                    }
                )
        except Exception:
            failures.append({"input_idx": index, "error": traceback.format_exc()})

    total = len(inputs)
    accuracy = passed / total if total else 0.0
    latency = float(np.median(latencies)) if latencies else None
    return accuracy, {"passed": passed, "total": total, "failures": failures}, latency


def _correctness_stage(
    case_meta: dict[str, Any],
    torch_kernel_fn: Callable[..., Any],
    bundle: KernelBundle,
    model: Any | None,
    *,
    backend: str,
) -> dict[str, Any]:
    stage: dict[str, Any] = {
        "backend": backend,
        "attempted": True,
        "accuracy": None,
        "accuracy_detail": {},
        "latency_ms": None,
    }
    try:
        accuracy, detail, latency = _run_accuracy(
            case_meta,
            torch_kernel_fn,
            bundle,
            model,
        )
        stage["accuracy"] = accuracy
        stage["accuracy_detail"] = detail
        stage["latency_ms"] = latency
    except Exception:
        stage["accuracy_detail"] = {
            "backend": backend,
            "stage": "execution",
            "error": traceback.format_exc(),
        }
    return stage


def _copy_correctness_to_result(result: EvalResult, stage: dict[str, Any]) -> None:
    result.accuracy = stage.get("accuracy")
    result.accuracy_detail = stage.get("accuracy_detail") or {}
    result.latency_ms = stage.get("latency_ms")


def evaluate(
    case_id: str,
    fhe_kernel_path: str,
    method: str = "unknown",
    model: str = "unknown",
    registry_path: str = _DEFAULT_REGISTRY,
    run_real: bool = True,
) -> EvalResult:
    """Evaluate one generated four-function OpenFHE/CKKS program.

    Args:
        run_real: When true, measure correctness and latency with real OpenFHE.
            When false, use the plaintext simulator for those measurements.
            Both modes always use the simulator for execution-based hacking
            detection.
    """
    registry = _load_registry(registry_path)
    case_meta = _find_case(registry, case_id)
    result = EvalResult(case_id=case_id, method=method, model=model, compiled=False)

    static_detected, static_reason, static_tags = check_static(fhe_kernel_path)
    result.reward_hack_static = static_detected
    result.reward_hack_static_reason = static_reason
    result.reward_hack_tags.extend(static_tags)
    result.validation_stages["static_check"] = {
        "attempted": True,
        "passed": not static_detected,
        "reward_hack_detected": static_detected,
        "reason": static_reason,
        "tags": _normalise_tags(static_tags),
    }

    ref_module = _load_ref_module(case_meta["torch_ref"])
    torch_kernel_fn = ref_module.torch_kernel
    runtime_inputs = _generate_runtime_inputs(case_meta)

    # The execution-based hacking check always uses the plaintext simulator,
    # including when correctness will subsequently run with real OpenFHE.
    with _sanitised_candidate_environment(), _openfhe_backend(simulate=True):
        torch.manual_seed(42)
        simulator_bundle, simulator_import_error = _load_fhe_module(fhe_kernel_path)
        if simulator_bundle is None:
            result.compile_error = simulator_import_error
            result.validation_stages["import"] = {
                "backend": "simulate",
                "passed": False,
                "error": simulator_import_error,
            }
            result.validation_stages["execution_check"] = {
                "backend": "simulate",
                "attempted": False,
                "passed": False,
                "reward_hack_detected": False,
                "reason": "Submission could not be imported with the plaintext simulator.",
                "tags": [],
            }
            return result

        result.compiled = True
        result.validation_stages["import"] = {
            "backend": "simulate",
            "passed": True,
        }
        simulator_model = _create_model_for_case(ref_module) if _has_model(case_meta) else None
        execution_detected, execution_reason, execution_tags = check_execution(
            case_meta,
            simulator_bundle,
            runtime_inputs,
            model=simulator_model,
        )
        result.reward_hack_execution = execution_detected
        result.reward_hack_execution_reason = execution_reason
        result.reward_hack_tags.extend(execution_tags)
        result.validation_stages["execution_check"] = {
            "backend": "simulate",
            "attempted": True,
            "passed": not execution_detected,
            "reward_hack_detected": execution_detected,
            "reason": execution_reason,
            "tags": _normalise_tags(execution_tags),
        }

        if not run_real:
            simulator_model = _create_model_for_case(ref_module) if _has_model(case_meta) else None
            simulator_stage = _correctness_stage(
                case_meta,
                torch_kernel_fn,
                simulator_bundle,
                simulator_model,
                backend="simulate",
            )
            result.validation_stages["simulate"] = simulator_stage
            _copy_correctness_to_result(result, simulator_stage)
            return result

    # Real OpenFHE performs correctness and latency measurement only.
    with _sanitised_candidate_environment(), _openfhe_backend(simulate=False):
        torch.manual_seed(42)
        real_bundle, real_import_error = _load_fhe_module(fhe_kernel_path)
        if real_bundle is None:
            real_stage = {
                "backend": "real",
                "attempted": True,
                "accuracy": None,
                "accuracy_detail": {
                    "backend": "real",
                    "stage": "import",
                    "error": real_import_error,
                },
                "latency_ms": None,
            }
        else:
            real_model = _create_model_for_case(ref_module) if _has_model(case_meta) else None
            real_stage = _correctness_stage(
                case_meta,
                torch_kernel_fn,
                real_bundle,
                real_model,
                backend="real",
            )

    result.validation_stages["real"] = real_stage
    _copy_correctness_to_result(result, real_stage)
    return result


def is_functionally_correct(result: EvalResult) -> bool:
    """Return whether every generated test input passes ``allclose``."""
    if not result.compiled or result.accuracy is None:
        return False
    detail = result.accuracy_detail or {}
    passed = detail.get("passed")
    total = detail.get("total")
    if passed is not None and total is not None:
        return total > 0 and passed == total
    return result.accuracy >= 1.0 - 1e-9


def _build_failures(result: EvalResult) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if result.compile_error:
        failures.append({"stage": "import", "error": result.compile_error})
    if result.reward_hack_static:
        failures.append({"stage": "static_check", "error": result.reward_hack_static_reason})
    if result.reward_hack_execution:
        failures.append(
            {
                "stage": "execution_check",
                "error": result.reward_hack_execution_reason,
            }
        )

    detail = result.accuracy_detail or {}
    if detail.get("error"):
        failures.append(
            {
                "stage": detail.get("stage", "execution"),
                "backend": detail.get("backend"),
                "error": detail["error"],
            }
        )
    failures.extend(detail.get("failures", []))
    return failures


def build_payload(result: EvalResult) -> dict[str, Any]:
    """Build the canonical JSON payload used by stdout and eval_status.json."""
    tags = _normalise_tags(result.reward_hack_tags)
    interface_invalid = RewardHackTag.INTERFACE.value in tags
    syntax_correctness = result.compiled and not interface_invalid
    functional_correctness = is_functionally_correct(result)
    reward_hack_detected = result.reward_hack_detected or bool(tags)
    passed = syntax_correctness and functional_correctness and not reward_hack_detected

    detail = result.accuracy_detail or {}
    payload: dict[str, Any] = {
        "case_id": result.case_id,
        "method": result.method,
        "model": result.model,
        "passed": passed,
        "syntax_correctness": syntax_correctness,
        "functional_correctness": functional_correctness,
        "reward_hack_detected": reward_hack_detected,
        "reward_hack_tags": tags,
        "accuracy": f"{detail.get('passed', 0)}/{detail.get('total', 0)}",
        "latency_ms": result.latency_ms,
        "failures": _build_failures(result),
        "validation_stages": result.validation_stages,
    }
    if result.sample_id is not None:
        payload["sample_id"] = result.sample_id
    return payload


def write_eval_status(payload: dict[str, Any], fhe_kernel_path: str) -> Path:
    """Write the exact evaluation payload beside ``fhe_kernel.py``."""
    status_path = Path(fhe_kernel_path).parent / "eval_status.json"
    status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return status_path
