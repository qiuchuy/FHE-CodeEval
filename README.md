# FHE-CodeEval

FHE-CodeEval evaluates whether large language models can translate executable
PyTorch specifications into functionally equivalent OpenFHE/CKKS programs. The
public release contains the 46 cases, prompt template, generation runner,
two-stage hacking detector, evaluator, and report generator used by the paper.

## Benchmark taxonomy

The public case IDs and directories use the paper's category and case names.

| Category | Cases | Case names |
| --- | ---: | --- |
| Packing-Oriented Operators | 20 | `matmul`, `double-matmul`, `ttm`, `convolution`, `logreg-matvecmul`, `dot-product`, `hamming-dist`, `l2-distance`, `lin-reg`, `poly-reg`, `poly-derivative`, `dct`, `box-blur`, `gx-kernel`, `gy-kernel`, `sobel`, `roberts-cross`, `sum`, `mean`, `avgpool` |
| Approximation-Oriented Operators | 14 | `relu`, `gelu`, `sigmoid`, `silu`, `exp`, `sqrt`, `sign`, `min`, `softmax`, `layernorm`, `batchnorm`, `tanh`, `log`, `inverse` |
| Compositional subgraphs | 7 | `linear-relu`, `conv2d-relu`, `linear-square`, `conv2d-square`, `linear-softmax`, `sum-relu`, `sum-square` |
| Neural-network workloads | 5 | `mlp`, `bert-attention`, `lenet5`, `resnet20`, `squeezenet` |

Each reference is stored at:

```text
benchmarks/<paper-category>/<case-name>/ref.py
```

The registry at `fhe_codeeval/registry/benchmarks.yaml` defines input shapes,
roles, ranges, tolerances, and deterministic sampling metadata.

## Repository layout

```text
benchmarks/                         46 PyTorch reference cases
configs/template.yaml               documented run configuration
fhe_codeeval/
  harness/                          data generation, evaluator, reports, hacking checks
  llm/                              Anthropic/OpenAI-compatible clients
  prompts/templates/prompt.md.j2    canonical paper prompt template
  registry/benchmarks.yaml          benchmark metadata
packages/simulate-ckks/             plaintext OpenFHE/CKKS simulator
run_benchmark.py                    batch generation and evaluation
evaluate.py                         evaluate one generated program
report.py                           rebuild aggregate run reports
```

Generated kernels, trajectories, and reports are written under `runs/` and are
ignored by Git.

## Requirements and installation

The paper's real-OpenFHE environment uses Ubuntu 22.04 x86_64 and Python 3.10. Install
[`uv`](https://docs.astral.sh/uv/), clone the repository, and create the locked
environment:

```bash
git clone https://github.com/qiuchuy/FHE-CodeEval.git
cd FHE-CodeEval
uv sync --locked
```

FHE-CodeEval is intentionally a clone-and-run `uv` project rather than a
standalone Python wheel: the runner, references, registry, and local simulator
are versioned together.

The lock file installs CPU-only PyTorch `2.5.0` and the other pinned
dependencies; OpenFHE `1.5.1.0.22.4` is installed only on Linux x86_64. Real
OpenFHE execution is intended for Ubuntu 22.04 x86_64. On macOS with Apple
Silicon or on Linux ARM, use simulator-only evaluation with `simulate: true`
or `--simulate`. The current lock file does not support Intel macOS.

## Configure a run

Keep local configs and credentials out of Git:

```bash
mkdir -p configs/local
cp configs/template.yaml configs/local/my-run.yaml
```

Edit `configs/local/my-run.yaml`. The main fields are:

| Field | Meaning |
| --- | --- |
| `method` | `one_shot`, `feedback`, or `agentic` |
| `model` | Human-readable model label saved in reports |
| `model_provider` | `anthropic`, `openai`, or a compatible alias |
| `model_id` | Exact provider model/API identifier |
| `model_base_url` / `_env` | Optional compatible endpoint and its environment variable |
| `model_api_key_env` | Environment variable containing the API key |
| `model_max_tokens`, `model_temperature` | Generation controls |
| `model_enable_thinking`, `model_extra_params` | Optional provider-specific controls |
| `model_use_max_completion_tokens` | Send the output cap as `max_completion_tokens` |
| `agentic_timeout_seconds` | Claude Code wall-clock budget per trial; the paper uses 5400 seconds |
| `cases` | Exact registry IDs; `[]` selects all 46 cases |
| `sampling_num` | Independent rollouts per model-case setting |
| `feedback_rounds` | Repair rounds after the initial generation |
| `num_workers` | Parallel model-case workers |
| `simulate` | Use simulator correctness/latency instead of real OpenFHE |
| `dry_run` | Render prompts without calling a model |
| `no_eval` | Generate programs without evaluating them |
| `setup_only` | Create agentic workspaces without launching Claude Code |
| `skip_completed` | Resume without repeating finished samples |

The checked-in template uses one real case and contains no secret. Export the
credential named by `model_api_key_env`, for example:

```bash
export ANTHROPIC_API_KEY='...'
```

For an OpenAI-compatible endpoint, use fields such as:

```yaml
method: feedback
model: my-open-model
model_provider: openai-compatible
model_id: provider/model-id
model_base_url: http://HOST:PORT/v1
model_base_url_env: OPENAI_BASE_URL
model_api_key: null
model_api_key_env: OPENAI_API_KEY
model_max_tokens: 32000
model_temperature: 0
model_extra_params: {}
```

The paper uses five independent rollouts (`sampling_num: 5`). Its iterative
setting permits five repairs after the initial generation
(`feedback_rounds: 5`).

## Run from the command line

Run the config:

```bash
uv run python run_benchmark.py --config configs/local/my-run.yaml
```

Command-line boolean flags and their inverse forms override config values; for
example, `--simulate`/`--no-simulate` and `--no-eval`/`--eval`. When changing
models or providers, update `model`, `model_provider`, `model_id`, and the
corresponding environment-variable fields together. For example, preview one
canonical prompt without making an API request:

```bash
uv run python run_benchmark.py \
  --config configs/local/my-run.yaml \
  --cases packing-oriented-operators/matmul \
  --sampling-num 1 \
  --dry-run
```

Resume a partially completed run:

```bash
uv run python run_benchmark.py \
  --config configs/local/my-run.yaml \
  --run-id <existing-run-id> \
  --skip-completed
```

Agentic mode requires the Claude Code CLI. It creates one isolated workspace
per case/sample containing `CLAUDE.md`, `prompt.txt`, `torch_ref.py`, and the
generated `fhe_kernel.py`.

## Evaluate one generated program

Generated programs must define the paper's four functions:

```python
def make_context(): ...
def encrypt(cc, keys, ...): ...
def fhe_kernel(cc, keys, enc_inputs): ...
def decrypt(cc, keys, ct_out): ...
```

Run the real OpenFHE correctness/latency stage:

```bash
uv run python evaluate.py \
  --case packing-oriented-operators/matmul \
  --fhe-kernel runs/<run-id>/matmul/1/fhe_kernel.py
```

Run correctness and latency with the plaintext simulator instead:

```bash
uv run python evaluate.py \
  --case packing-oriented-operators/matmul \
  --fhe-kernel runs/<run-id>/matmul/1/fhe_kernel.py \
  --simulate
```

Both commands always apply the paper's two hacking-detection stages:

1. **Static checking** verifies the four-function interface and rejects
   forbidden encryption/decryption/plaintext-extraction APIs and PyTorch/NumPy
   fallbacks reachable from `fhe_kernel`.
2. **Execution-based checking** uses the plaintext OpenFHE simulator to confirm
   homomorphic compute calls and `torch.profiler` to detect case-specific
   PyTorch execution.

A candidate passes only when it imports, passes all numerical checks, and is
not flagged by either hacking-detection stage. Only `fhe_kernel` latency is
timed.

## Reports

Every evaluated case/sample writes `eval_status.json`; feedback and agentic
runs also write their conversation/trajectory metadata. Rebuild `report.json`
and `report.md` with:

```bash
uv run python report.py runs/<run-id>
```

The report includes functional/hacking summaries, latency, and unbiased
Pass@1/Pass@5 when enough samples are available.

Each non-dry run also writes a credential-free `run_config.json` containing
the resolved model settings, selected cases, method, sampling/feedback budget,
backend, and Git revision.

## Safety

Generated Python and agentic coding tools are untrusted code. Run this
benchmark only in a disposable container, virtual machine, or dedicated host
without unrelated credentials or sensitive files. Do not place API keys in
YAML; expose only the credential required for the current generation process.
Agentic mode launches Claude Code with non-interactive permissions inside its
case workspace, so host-level isolation is still required.

On a Linux host managed by systemd, an optional memory cap can be applied to a
run, adjusted to the machine's available RAM:

```bash
systemd-run --user --scope -p MemoryMax=32G -- \
  uv run python run_benchmark.py --config configs/local/my-run.yaml
```

## Operational note

LLMs may generate highly inefficient OpenFHE/CKKS programs. On
memory-constrained hosts, limit per-process memory according to available
system resources.

## License

FHE-CodeEval is released under the [MIT License](LICENSE). The bundled
plaintext simulator retains the OpenFHE [BSD 2-Clause License](packages/simulate-ckks/LICENSE).
