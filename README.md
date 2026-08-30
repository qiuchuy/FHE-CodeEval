# FHE-CodeEval

FHE-CodeEval is a collection of small PyTorch reference kernels for evaluating
code generation and compilation workflows for fully homomorphic encryption
(FHE). Each benchmark isolates one tensor operation behind a common
`torch_kernel` entry point.

## Repository layout

```text
benchmarks/
├── tier1/              # Primitive tensor operations
│   └── <category>/<operation>/torch_/ref.py
└── tier2/              # Composed operations
    └── <category>/<operation>/torch_/ref.py
```

Tier 1 currently covers convolution, data arrangement, dot products,
element-wise operations, nonlinear functions, normalization, pooling,
reductions, and view operations. Tier 2 currently contains multi-head
attention.

## Using a reference kernel

Install PyTorch using the instructions appropriate for your platform, then load
the desired `ref.py`. For example:

```python
import importlib.util
from pathlib import Path

path = Path("benchmarks/tier1/elementary/add/torch_/ref.py")
spec = importlib.util.spec_from_file_location("add_reference", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

result = module.torch_kernel(a, b)
```

Input shapes and dtypes are determined by the evaluation harness. Reference
kernels should remain small and deterministic, and must expose a top-level
function named `torch_kernel`.

## Development

The maintenance checks require Python 3.10 or newer and
[Ruff](https://docs.astral.sh/ruff/):

```bash
python -m pip install ruff
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before adding or changing a benchmark.

## License

This project is licensed under the [MIT License](LICENSE).
