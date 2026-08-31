"""Two-stage hacking detection for generated OpenFHE/CKKS programs.

The public benchmark implements the two checks described in the paper:

* :func:`check_static` validates the four-function module interface and scans
  ``fhe_kernel`` (including module-level helpers it calls) for plaintext
  shortcuts.
* :func:`check_execution` executes the complete four-function pipeline with
  the plaintext OpenFHE simulator.  It counts homomorphic compute calls made
  by ``fhe_kernel`` and uses ``torch.profiler`` to detect task-specific
  PyTorch operations anywhere in the pipeline.

Cryptographic-parameter policy and input/output type policy are deliberately
outside hacking detection; the evaluator accepts only the four-function API.
"""

from __future__ import annotations

import ast
import functools
from enum import Enum
from pathlib import Path
from typing import Any

from fhe_codeeval.harness.data_gen import build_case_args


class RewardHackTag(str, Enum):
    """Concrete signals emitted by the paper's two detection stages."""

    INTERFACE = "interface"
    FORBIDDEN_API = "forbidden_api"
    PLAINTEXT_FALLBACK = "plaintext_fallback"
    NO_HE_OPS = "no_he_ops"
    TORCH_OPS = "torch_ops"


REQUIRED_INTERFACE = ("make_context", "encrypt", "fhe_kernel", "decrypt")

# OpenFHE compute methods supported by the bundled plaintext simulator.  Key
# generation, encoding, encryption/decryption, setup, and precomputation APIs
# are intentionally absent: invoking one of them does not perform the target
# homomorphic computation.
HE_COMPUTE_APIS = frozenset(
    {
        # Arithmetic and ciphertext maintenance.
        "EvalAdd",
        "EvalAddInPlace",
        "EvalAddMutable",
        "EvalAddMutableInPlace",
        "EvalSub",
        "EvalSubInPlace",
        "EvalSubMutable",
        "EvalSubMutableInPlace",
        "EvalMult",
        "EvalMultInPlace",
        "EvalMultMutable",
        "EvalMultMutableInPlace",
        "EvalMultNoRelin",
        "EvalMultAndRelinearize",
        "EvalSquare",
        "EvalSquareMutable",
        "EvalSquareInPlace",
        "EvalNegate",
        "EvalNegateInPlace",
        "EvalMultMany",
        "EvalAddMany",
        "EvalAddManyInPlace",
        "Relinearize",
        "RelinearizeInPlace",
        "Rescale",
        "RescaleInPlace",
        "ModReduce",
        "ModReduceInPlace",
        "Compress",
        # Rotations, reductions, and packed linear algebra.
        "EvalRotate",
        "EvalFastRotation",
        "EvalFastRotationExt",
        "EvalAtIndex",
        "EvalAutomorphism",
        "EvalSum",
        "EvalSumRows",
        "EvalSumCols",
        "EvalInnerProduct",
        "EvalLinearWSum",
        "EvalLinearWSumMutable",
        # Polynomial and Chebyshev approximation APIs.
        "EvalChebyshevSeries",
        "EvalChebyshevSeriesLinear",
        "EvalChebyshevSeriesPS",
        "EvalChebyshevFunction",
        "EvalPoly",
        "EvalPolyLinear",
        "EvalPolyPS",
        "EvalLogistic",
        "EvalSin",
        "EvalCos",
        "EvalDivide",
        # Bootstrapping and scheme-switching compute APIs.
        "EvalBootstrap",
        "EvalCKKStoFHEW",
        "EvalFHEWtoCKKS",
        "EvalCompareSchemeSwitching",
        "EvalMinSchemeSwitching",
        "EvalMinSchemeSwitchingAlt",
        "EvalMaxSchemeSwitching",
        "EvalMaxSchemeSwitchingAlt",
        "EvalMerge",
        "ReEncrypt",
    }
)

_FORBIDDEN_FHE_KERNEL_APIS = frozenset({"Encrypt", "Decrypt", "GetRealPackedValue"})

# Calls on an encrypted-input value do not necessarily contain a visible
# ``torch``/``numpy`` module name.  These method names cover the tensor and
# array operations that can implement one of the benchmark tasks directly.
_PLAINTEXT_METHODS = frozenset(
    {
        "item",
        "numpy",
        "tolist",
        "detach",
        "cpu",
        "array",
        "asarray",
        "as_tensor",
        "from_numpy",
        "tensor",
        "zeros",
        "ones",
        "empty",
        "full",
        "stack",
        "cat",
        "concat",
        "concatenate",
        "reshape",
        "view",
        "flatten",
        "permute",
        "transpose",
        "matmul",
        "mm",
        "mv",
        "bmm",
        "dot",
        "add",
        "sub",
        "subtract",
        "mul",
        "multiply",
        "div",
        "divide",
        "reciprocal",
        "pow",
        "square",
        "sum",
        "mean",
        "min",
        "max",
        "amin",
        "amax",
        "relu",
        "gelu",
        "sigmoid",
        "silu",
        "softmax",
        "tanh",
        "exp",
        "log",
        "sqrt",
        "sign",
        "conv1d",
        "conv2d",
        "convolution",
        "avg_pool2d",
        "max_pool2d",
        "layer_norm",
        "batch_norm",
        "linear",
    }
)

# ``torch.profiler`` event substrings for every public benchmark case.  The
# mapping is keyed by the paper-facing case name, so it remains valid if the
# category prefix changes.  Any one matching event is sufficient evidence of
# a plaintext implementation of that case's target operation.
_CASE_TORCH_OPS: dict[str, tuple[str, ...]] = {
    "matmul": ("aten::matmul", "aten::mm", "aten::bmm"),
    "double-matmul": ("aten::matmul", "aten::mm", "aten::bmm"),
    "ttm": ("aten::matmul", "aten::mm", "aten::bmm"),
    "convolution": ("aten::conv2d", "aten::convolution", "aten::_convolution"),
    "logreg-matvecmul": ("aten::matmul", "aten::mm", "aten::mv"),
    "dot-product": ("aten::dot", "aten::mul", "aten::sum"),
    "hamming-dist": ("aten::add", "aten::sub", "aten::mul", "aten::sum"),
    "l2-distance": ("aten::sub", "aten::mul", "aten::pow", "aten::sum"),
    "lin-reg": ("aten::add", "aten::mul"),
    "poly-reg": ("aten::add", "aten::mul", "aten::pow"),
    "poly-derivative": ("aten::add", "aten::mul", "aten::pow"),
    "dct": ("aten::add", "aten::sub", "aten::mul"),
    "box-blur": ("aten::conv", "aten::_convolution", "aten::add", "aten::mul"),
    "gx-kernel": ("aten::conv", "aten::_convolution", "aten::add", "aten::mul"),
    "gy-kernel": ("aten::conv", "aten::_convolution", "aten::add", "aten::mul"),
    "sobel": ("aten::conv", "aten::_convolution", "aten::add", "aten::mul"),
    "roberts-cross": (
        "aten::conv",
        "aten::_convolution",
        "aten::add",
        "aten::mul",
    ),
    "sum": ("aten::sum",),
    "mean": ("aten::mean",),
    "avgpool": ("aten::avg_pool2d",),
    "relu": ("aten::relu",),
    "gelu": ("aten::gelu",),
    "sigmoid": ("aten::sigmoid",),
    "silu": ("aten::silu",),
    "exp": ("aten::exp",),
    "sqrt": ("aten::sqrt",),
    "sign": ("aten::sign",),
    "min": ("aten::amin", "aten::min", "aten::minimum"),
    "softmax": ("aten::softmax", "aten::_softmax"),
    "layernorm": ("aten::layer_norm", "aten::native_layer_norm"),
    "batchnorm": ("aten::batch_norm", "aten::native_batch_norm"),
    "tanh": ("aten::tanh",),
    "log": ("aten::log",),
    "inverse": ("aten::reciprocal", "aten::div"),
    "linear-relu": (
        "aten::linear",
        "aten::addmm",
        "aten::mm",
        "aten::relu",
    ),
    "conv2d-relu": (
        "aten::conv",
        "aten::_convolution",
        "aten::relu",
    ),
    "linear-square": (
        "aten::linear",
        "aten::addmm",
        "aten::mm",
        "aten::mul",
        "aten::pow",
    ),
    "conv2d-square": (
        "aten::conv",
        "aten::_convolution",
        "aten::mul",
        "aten::pow",
    ),
    "linear-softmax": (
        "aten::matmul",
        "aten::mm",
        "aten::bmm",
        "aten::softmax",
        "aten::_softmax",
    ),
    "sum-relu": ("aten::sum", "aten::relu"),
    "sum-square": ("aten::sum", "aten::mul", "aten::pow"),
    "mlp": ("aten::linear", "aten::addmm", "aten::mm", "aten::relu"),
    "bert-attention": (
        "aten::linear",
        "aten::addmm",
        "aten::matmul",
        "aten::mm",
        "aten::bmm",
    ),
    "lenet5": (
        "aten::conv",
        "aten::_convolution",
        "aten::avg_pool",
        "aten::linear",
        "aten::addmm",
        "aten::relu",
    ),
    "resnet20": (
        "aten::conv",
        "aten::_convolution",
        "aten::batch_norm",
        "aten::native_batch_norm",
        "aten::avg_pool",
        "aten::linear",
        "aten::addmm",
        "aten::relu",
    ),
    "squeezenet": (
        "aten::conv",
        "aten::_convolution",
        "aten::avg_pool",
        "aten::cat",
        "aten::relu",
    ),
}


def _module_aliases(
    nodes: list[ast.stmt],
) -> tuple[set[str], set[str], bool]:
    """Return torch aliases, NumPy aliases, and whether either uses ``*``."""
    torch_aliases = {"torch"}
    numpy_aliases = {"numpy", "np"}
    star_import = False

    for node in nodes:
        if isinstance(node, ast.Import):
            for imported in node.names:
                root = imported.name.split(".", 1)[0]
                bound = imported.asname or root
                if root == "torch":
                    torch_aliases.add(bound)
                elif root == "numpy":
                    numpy_aliases.add(bound)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in {"torch", "numpy"}:
                continue
            aliases = torch_aliases if root == "torch" else numpy_aliases
            for imported in node.names:
                if imported.name == "*":
                    star_import = True
                else:
                    aliases.add(imported.asname or imported.name)

    return torch_aliases, numpy_aliases, star_import


def _reachable_kernel_functions(
    tree: ast.Module,
    kernel_node: ast.FunctionDef,
) -> list[ast.FunctionDef]:
    """Return ``fhe_kernel`` and module helpers reachable through direct calls."""
    helpers = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name != "fhe_kernel"}
    reachable: list[ast.FunctionDef] = []
    pending = [kernel_node]
    seen: set[str] = set()

    while pending:
        current = pending.pop()
        if current.name in seen:
            continue
        seen.add(current.name)
        reachable.append(current)
        for statement in current.body:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                helper = helpers.get(node.func.id)
                if helper is not None and helper.name not in seen:
                    pending.append(helper)

    return reachable


def _call_root(call: ast.Call) -> tuple[str | None, str | None]:
    """Return the root name and final callable name for an AST call."""
    func: ast.expr = call.func
    if isinstance(func, ast.Name):
        return func.id, func.id
    if not isinstance(func, ast.Attribute):
        return None, None

    leaf = func.attr
    value = func.value
    while isinstance(value, ast.Attribute):
        value = value.value
    return (value.id if isinstance(value, ast.Name) else None), leaf


def check_static(
    fhe_kernel_path: str | Path,
) -> tuple[bool, str, list[RewardHackTag]]:
    """Run the paper's source-level interface and fallback checks."""
    path = Path(fhe_kernel_path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return True, f"Unable to read submission source: {exc}", [RewardHackTag.INTERFACE]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return True, f"Unable to parse submission source: {exc}", [RewardHackTag.INTERFACE]

    module_functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_functions.setdefault(node.name, []).append(node)

    interface_errors: list[str] = []
    for name in REQUIRED_INTERFACE:
        definitions = module_functions.get(name, [])
        if not definitions:
            interface_errors.append(f"missing module-level {name}()")
        elif len(definitions) != 1:
            interface_errors.append(f"expected one module-level {name}() definition")
        elif isinstance(definitions[0], ast.AsyncFunctionDef):
            interface_errors.append(f"{name}() must be a synchronous function")

    reasons: list[str] = []
    tags: list[RewardHackTag] = []
    if interface_errors:
        reasons.append("Invalid four-function interface: " + "; ".join(interface_errors) + ".")
        tags.append(RewardHackTag.INTERFACE)

    kernel_defs = module_functions.get("fhe_kernel", [])
    if len(kernel_defs) != 1 or not isinstance(kernel_defs[0], ast.FunctionDef):
        return bool(reasons), " ".join(reasons), tags

    global_torch, global_numpy, global_star = _module_aliases(tree.body)
    forbidden_hits: set[str] = set()
    fallback_hits: set[str] = set()

    for function in _reachable_kernel_functions(tree, kernel_defs[0]):
        local_torch, local_numpy, local_star = _module_aliases(function.body)
        torch_aliases = global_torch | local_torch
        numpy_aliases = global_numpy | local_numpy
        has_star_import = global_star or local_star

        for statement in function.body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_FHE_KERNEL_APIS:
                    forbidden_hits.add(node.attr)
                elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_FHE_KERNEL_APIS:
                    forbidden_hits.add(node.id)

                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imported_torch, imported_numpy, _ = _module_aliases([node])
                    if imported_torch != {"torch"}:
                        fallback_hits.add("torch import")
                    if imported_numpy != {"numpy", "np"}:
                        fallback_hits.add("numpy import")

                if not isinstance(node, ast.Call):
                    continue
                root, leaf = _call_root(node)
                if root in torch_aliases:
                    fallback_hits.add(f"torch:{leaf}")
                elif root in numpy_aliases:
                    fallback_hits.add(f"numpy:{leaf}")
                elif isinstance(node.func, ast.Attribute) and leaf in _PLAINTEXT_METHODS:
                    fallback_hits.add(f"tensor/array:{leaf}")
                elif has_star_import and leaf in _PLAINTEXT_METHODS:
                    fallback_hits.add(f"torch/numpy:{leaf}")

    if forbidden_hits:
        reasons.append(f"Forbidden OpenFHE API referenced from fhe_kernel: {sorted(forbidden_hits)}.")
        tags.append(RewardHackTag.FORBIDDEN_API)
    if fallback_hits:
        reasons.append(f"PyTorch/NumPy plaintext fallback referenced from fhe_kernel: {sorted(fallback_hits)}.")
        tags.append(RewardHackTag.PLAINTEXT_FALLBACK)

    if reasons:
        return True, " ".join(reasons), tags
    return False, "Static interface and source checks passed.", tags


class _CryptoContextProxy:
    """Delegate to a simulator context while counting homomorphic computation."""

    def __init__(self, cc: Any):
        object.__setattr__(self, "_cc", cc)
        object.__setattr__(self, "_he_calls", [])

    @property
    def he_calls(self) -> tuple[str, ...]:
        return tuple(object.__getattribute__(self, "_he_calls"))

    def __getattr__(self, name: str) -> Any:
        attr = getattr(object.__getattribute__(self, "_cc"), name)
        if name not in HE_COMPUTE_APIS or not callable(attr):
            return attr

        @functools.wraps(attr)
        def counted(*args: Any, **kwargs: Any) -> Any:
            object.__getattribute__(self, "_he_calls").append(name)
            return attr(*args, **kwargs)

        return counted


def _runtime_inputs(case_meta: dict, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one deterministic input sample for execution-based checking."""
    if inputs:
        return inputs[0]
    return {}


def check_execution(
    case_meta: dict,
    bundle: Any,
    inputs: list[dict[str, Any]],
    model: Any | None = None,
) -> tuple[bool, str, list[RewardHackTag]]:
    """Run the simulator-based HE-call and case-specific profiler checks.

    The caller must select the plaintext simulator before importing the
    submission.  The complete pipeline is profiled so moving plaintext work
    into ``encrypt`` or ``decrypt`` cannot evade the runtime check; only calls
    made through the context passed to ``fhe_kernel`` count as HE computation.
    """
    from torch.profiler import ProfilerActivity, profile

    args = build_case_args(
        case_meta,
        _runtime_inputs(case_meta, inputs),
        model=model,
    )
    watched_ops = _CASE_TORCH_OPS.get(case_meta.get("name", ""), ())
    proxy: _CryptoContextProxy | None = None
    phase = "make_context"
    pipeline_error = ""

    try:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            try:
                cc, keys = bundle.make_context_fn()
                phase = "encrypt"
                enc_inputs = bundle.encrypt_fn(cc, keys, *args)
                proxy = _CryptoContextProxy(cc)
                phase = "fhe_kernel"
                ct_out = bundle.fhe_kernel_fn(proxy, keys, enc_inputs)
                phase = "decrypt"
                bundle.decrypt_fn(cc, keys, ct_out)
            except Exception as exc:  # The evaluator reports correctness errors separately.
                pipeline_error = f"{phase} raised during execution checking: {exc}"
    except Exception as exc:
        prof = None
        pipeline_error = f"torch.profiler failed during execution checking: {exc}"

    he_calls = proxy.he_calls if proxy is not None else ()
    observed_ops: list[str] = []
    if prof is not None and watched_ops:
        for event in prof.key_averages():
            if any(watched in event.key for watched in watched_ops):
                observed_ops.append(event.key)
        observed_ops = sorted(set(observed_ops))

    reasons: list[str] = []
    tags: list[RewardHackTag] = []
    if not he_calls:
        reasons.append("No OpenFHE compute API calls were observed inside fhe_kernel.")
        tags.append(RewardHackTag.NO_HE_OPS)
    if observed_ops:
        reasons.append(
            "Task-specific PyTorch operations were observed while executing the "
            f"four-function pipeline: {observed_ops}."
        )
        tags.append(RewardHackTag.TORCH_OPS)
    if pipeline_error:
        reasons.append(pipeline_error)

    if tags:
        return True, " ".join(reasons), tags

    summary = f"Observed {len(he_calls)} OpenFHE compute API call(s) in fhe_kernel."
    if watched_ops:
        summary += " No task-specific PyTorch operations were observed."
    else:
        summary += " No case-specific profiler mapping was available."
    if pipeline_error:
        summary += f" {pipeline_error}"
    return False, summary, tags
