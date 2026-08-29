# Contributing

## Adding a benchmark

1. Place the reference at
   `benchmarks/<tier>/<category>/<operation>/torch_/ref.py`.
2. Define exactly one top-level `torch_kernel` function in that file.
3. Keep the kernel deterministic and free of file, network, and global-state
   side effects.
4. Use tensor arguments rather than generating inputs inside the kernel.
5. Document any fixed parameters (such as stride, padding, or reduction axis)
   in the pull request.

Run the repository checks before submitting a change:

```bash
ruff check .
ruff format --check .
python -m unittest discover -s tests -v
```

Behavioral changes to existing references should include the reason for the
change and the expected input and output shapes. Avoid bundling unrelated
benchmark changes in the same commit.
