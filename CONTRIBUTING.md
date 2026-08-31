# Contributing

FHE-CodeEval's public benchmark taxonomy is fixed to the 46 cases described in
the paper. Changes to case names, categories, input metadata, or reference
semantics should therefore be discussed before opening a pull request.

## Editing an existing case

- References live at `benchmarks/<paper-category>/<case-name>/ref.py`.
- Keep exactly one module-level `torch_kernel` entry point.
- Keep `input_specs` in `fhe_codeeval/registry/benchmarks.yaml` in the same
  order as the function arguments.
- Preserve `seed_id` when renaming a public ID so deterministic evaluation
  inputs do not change.
- Use `sampling_strategy: uniform` or `stratified` explicitly.
- Do not commit generated kernels, run reports, trajectories, credentials, or
  local configs.

Install the locked development environment and run the public checks:

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run python -m compileall -q run_benchmark.py evaluate.py report.py fhe_codeeval
uv run python run_benchmark.py \
  --config configs/template.yaml \
  --cases packing-oriented-operators/dot-product \
  --sampling-num 1 \
  --dry-run
```
