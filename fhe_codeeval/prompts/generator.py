"""Build the canonical FHE-CodeEval prompt for every evaluation method.

``one_shot`` and ``feedback`` return the rendered prompt as a string.
``agentic`` writes that same prompt to ``prompt.txt`` and creates the small
workspace used by the coding agent.
"""

from __future__ import annotations

import importlib.util
import shlex
from pathlib import Path

import yaml

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent
_REGISTRY_PATH = _PACKAGE_ROOT / "registry" / "benchmarks.yaml"
_TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"
_PROMPT_TEMPLATE = _TEMPLATES_ROOT / "prompt.md.j2"
_AGENTIC_TEMPLATE = _TEMPLATES_ROOT / "agentic" / "CLAUDE.md.j2"
_EXAMPLE_PATH = _TEMPLATES_ROOT / "example_problem_and_solution.md"


def _load_registry() -> list[dict]:
    with _REGISTRY_PATH.open(encoding="utf-8") as registry_file:
        registry = yaml.safe_load(registry_file)
    if not isinstance(registry, list):
        raise ValueError(f"Expected a list of cases in {_REGISTRY_PATH}")
    return registry


def _find_case(registry: list[dict], case_id: str) -> dict:
    for case in registry:
        if case["id"] == case_id:
            return case
    raise KeyError(f"Case {case_id!r} not found in registry")


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _read_torch_source(torch_ref_path: str | Path) -> str:
    return _project_path(torch_ref_path).read_text(encoding="utf-8")


def _format_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_range(value_range: list | tuple) -> str:
    lo, hi = value_range
    return f"[{_format_number(lo)}, {_format_number(hi)}]"


def _role_label(spec: dict) -> str:
    if spec.get("type") == "scalar":
        return "[scalar constant]"
    if spec.get("role", "ciphertext") == "plaintext":
        return "[plaintext: known to the server]"
    return "[ciphertext: encrypted client data]"


def _random_input_summary(case_meta: dict) -> str:
    """Describe the actual sample count and declared tensor ranges."""
    n_samples = case_meta.get("n_test_inputs", 5)
    ranges: list[tuple] = []
    for spec in case_meta.get("input_specs", []):
        if spec.get("type") == "scalar" or spec.get("name") == "model":
            continue
        value_range = tuple(spec.get("range", [-1, 1]))
        if value_range not in ranges:
            ranges.append(value_range)

    if len(ranges) == 1:
        return f"  {n_samples} random test inputs in {_format_range(ranges[0])}"
    return f"  {n_samples} random test inputs using the ranges listed below"


def _format_input_specs(case_meta: dict) -> str:
    accuracy = case_meta.get("accuracy", {})
    atol = _format_number(accuracy.get("atol", 1e-2))
    rtol = _format_number(accuracy.get("rtol", 1e-2))
    lines = [
        _random_input_summary(case_meta),
        f"  Accuracy metric: torch.allclose (atol={atol}, rtol={rtol})",
        "",
    ]

    for spec in case_meta.get("input_specs", []):
        name = spec["name"]
        role = _role_label(spec)
        if spec.get("type") == "scalar":
            lines.append(f"  - {name}: scalar = {_format_number(spec['value'])} {role}")
            continue
        if name == "model":
            lines.append(f"  - model: torch.nn.Module (pre-initialized) {role}")
            continue

        shape = spec["shape"]
        dtype = spec.get("dtype", "float32")
        value_range = _format_range(spec.get("range", [-1, 1]))
        lines.append(f"  - {name}: torch.Tensor shape={shape}, dtype={dtype}, range={value_range} {role}")

    return "\n".join(lines)


def _is_network_case(case_meta: dict) -> bool:
    return any(
        spec.get("name") == "model" and spec.get("role") == "plaintext" for spec in case_meta.get("input_specs", [])
    )


def _load_reference_module(case_meta: dict, module_name: str):
    ref_path = _project_path(case_meta["torch_ref"])
    spec = importlib.util.spec_from_file_location(module_name, ref_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PyTorch reference at {ref_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _format_activation_ranges(ranges: dict[str, tuple[float, float]]) -> str:
    if not ranges:
        return ""
    lines = [
        "Use these observed ranges as approximation domains for the corresponding nonlinear operations.",
        "",
        "| Layer | Observed range |",
        "| --- | --- |",
    ]
    for layer, (lo, hi) in ranges.items():
        lines.append(f"| `{layer}` | [{lo:.3f}, {hi:.3f}] |")
    return "\n".join(lines)


def _get_activation_ranges(case_meta: dict) -> str:
    """Profile the exact evaluation inputs for network cases when possible."""
    if not _is_network_case(case_meta):
        return ""

    try:
        import torch

        from fhe_codeeval.harness.data_gen import generate_inputs
        from fhe_codeeval.harness.range_profiler import profile_activation_ranges

        module = _load_reference_module(case_meta, "fhe_codeeval_prompt_reference")
        if not hasattr(module, "create_model"):
            return ""

        torch.manual_seed(42)
        model = module.create_model()
        runtime_specs = [spec for spec in case_meta.get("input_specs", []) if spec.get("name") != "model"]
        eval_meta = {**case_meta, "input_specs": runtime_specs}
        eval_inputs = generate_inputs(eval_meta)
        ranges = profile_activation_ranges(
            case_meta,
            module.torch_kernel,
            model,
            inputs=eval_inputs,
        )
        return _format_activation_ranges(ranges)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        # Activation ranges are supplementary. Prompt generation must remain
        # available on machines that have not installed the heavyweight FHE
        # and PyTorch runtime yet.
        return ""


def _render_jinja(template_path: Path, context: dict) -> str:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    environment = Environment(
        loader=FileSystemLoader(_TEMPLATES_ROOT),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template_name = template_path.relative_to(_TEMPLATES_ROOT).as_posix()
    return environment.get_template(template_name).render(**context).strip() + "\n"


def _build_prompt_context(case_meta: dict) -> dict:
    return {
        "case_id": case_meta["id"],
        "task_description": case_meta.get("description", case_meta["name"]),
        "input_specification": _format_input_specs(case_meta),
        "activation_ranges": _get_activation_ranges(case_meta),
        "pytorch_reference": _read_torch_source(case_meta["torch_ref"]).rstrip(),
        "example_problem_and_solution": _EXAMPLE_PATH.read_text(encoding="utf-8").strip(),
    }


def _build_canonical_prompt(case_meta: dict) -> str:
    return _render_jinja(_PROMPT_TEMPLATE, _build_prompt_context(case_meta))


def _setup_workspace(
    case_meta: dict,
    run_id: str,
    sample_id: str | None = None,
    simulate: bool = False,
) -> Path:
    case_name = case_meta["id"].split("/")[-1]
    workspace = Path("runs") / run_id / case_name
    if sample_id is not None:
        workspace = workspace / str(sample_id)
    workspace.mkdir(parents=True, exist_ok=True)

    evaluator_args = [
        "uv",
        "run",
        "python",
        str(_PROJECT_ROOT / "evaluate.py"),
        "--case",
        case_meta["id"],
        "--fhe-kernel",
        str(workspace.resolve() / "fhe_kernel.py"),
    ]
    if simulate:
        evaluator_args.append("--simulate")
    evaluator_cmd = shlex.join(evaluator_args)
    agentic_context = {
        "evaluator_cmd": evaluator_cmd,
        "workspace_path": str(workspace.resolve()),
    }

    (workspace / "CLAUDE.md").write_text(
        _render_jinja(_AGENTIC_TEMPLATE, agentic_context),
        encoding="utf-8",
    )
    (workspace / "prompt.txt").write_text(
        _build_canonical_prompt(case_meta),
        encoding="utf-8",
    )
    (workspace / "torch_ref.py").write_text(
        _read_torch_source(case_meta["torch_ref"]),
        encoding="utf-8",
    )
    (workspace / "fhe_kernel.py").write_text(
        "# Write the generated OpenFHE/CKKS implementation here.\n",
        encoding="utf-8",
    )
    return workspace


def build_prompt_messages(
    case_id: str,
    method: str = "one_shot",
) -> list[dict[str, str]]:
    """Return the canonical prompt as one user message.

    The complete static and case-specific instructions intentionally live in
    one template, so no separate system message can drift from ``prompt.txt``.
    """
    if method not in {"one_shot", "feedback"}:
        raise ValueError(f"build_prompt_messages only supports one_shot/feedback, got {method!r}")
    case_meta = _find_case(_load_registry(), case_id)
    return [{"role": "user", "content": _build_canonical_prompt(case_meta)}]


def build_prompt(
    case_id: str,
    method: str,
    run_id: str = "run001",
    sample_id: str | None = None,
    simulate: bool = False,
) -> str | Path:
    """Build a prompt or agentic workspace for one registered case."""
    case_meta = _find_case(_load_registry(), case_id)
    if method in {"one_shot", "feedback"}:
        return _build_canonical_prompt(case_meta)
    if method == "agentic":
        return _setup_workspace(case_meta, run_id, sample_id, simulate=simulate)
    raise ValueError(f"Unknown method {method!r}. Choose: one_shot, feedback, agentic")
