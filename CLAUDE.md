# CLAUDE.md — Sophon

Sophon (`sophon-bench` on PyPI, import `sophon`, CLI `sophon`) is a generative
benchmark for spatial reasoning in VLMs. It ships **no images**: a YAML spec +
an integer seed deterministically produces `(image, question, ground_truth)`
triples. Two point sets (blue = reference, red = transformed copy) are rendered
in a coordinate system, and the model is asked about the transform between them.
See `README.md` for the science and figures.

## Setup

```bash
uv sync --all-extras          # dev + eval extras into .venv
uv run pytest                 # ~270 tests, ~25s, CPU only
uv run sophon --help
```

Python ≥ 3.11, managed with `uv`. `uv.lock` is committed; CI runs
`uv sync --locked`, so re-lock (`uv lock`) whenever `pyproject.toml` deps change.

## Common tasks

`--spec` accepts a file path **or** a built-in name (`cartesian_rotation` →
`sophon/specs/cartesian_rotation.yaml`).

```bash
# Look at a few samples (question + answer overlaid on the image) -> ./preview/
uv run sophon preview --spec cartesian_rotation --num-samples 5

# Generate a dataset to disk
uv run sophon generate --spec cartesian_rotation --seed 42 --output-dir out/rot
#   out/rot/images/000000.png ...   metadata.jsonl   ground_truth.json
#   (+ conversations.jsonl when the spec sets output_format: conversation)

# Whole suite (generate/evaluate accept the full_suite meta-spec; preview does not)
uv run sophon generate --spec full_suite --seed 42 --output-dir out/suite

# Evaluate a model over an OpenAI-compatible API (vllm, OpenRouter, OpenAI)
export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY; local vllm needs none
uv run sophon evaluate --spec full_suite --seeds 42,43,44 \
    --api-base https://openrouter.ai/api/v1 --model-name openai/gpt-4o \
    --output-dir results/gpt4o --max-rpm 60
#   -> <spec>[_seed<N>]_predictions.jsonl and <spec>[_seed<N>]_results.json (suffix only with multiple seeds)

# Re-score saved predictions
uv run sophon score --predictions results/x_predictions.jsonl --ground-truth out/rot/ground_truth.json
```

Python API:

```python
import yaml
from sophon.generator import BenchmarkGenerator
from sophon.schema import BenchmarkSpec

spec = BenchmarkSpec(**yaml.safe_load(open("sophon/specs/cartesian_rotation.yaml")))
for img, sample in BenchmarkGenerator(spec).generate(seed=42):
    ...  # img: PIL.Image | None, sample: sophon.schema.Sample
```

**Seed convention:** evaluation seeds are `42, 43, 44`; training/fine-tuning
data uses seeds `≥ 100` so train and eval never overlap.

**Sample count:** `num_samples` is per question type. A spec with
`num_samples: 200` and three `questions` entries yields 600 samples.

## Writing a spec

Copy a file in `sophon/specs/` and edit it. `sophon/schema.py` is the source of
truth; all models use `extra="forbid"`, so typos fail loudly.

```yaml
name: my_spec
coordinate_system: cartesian_2d     # linear_1d | cartesian_2d | polar | all
points: {num_points: [30, 60], distribution: uniform, margin: 0.1}
transforms:                         # name -> [min, max]; omit to disable
  rotation_deg: [-45, 45]
questions:
  - type: binary                    # binary | categorical | comparative | quantitative
    alignment_threshold: 0.05
  - type: categorical               # vocab_mode: restricted|full, vocab_order: fixed|random|random_correct_last
  - type: quantitative              # range_hint: true adds the valid range to the prompt
num_samples: 200
seed: 42
balance: random                     # exact = stratified label balance
output_mode: image                  # image | text_description | both
output_format: standard             # standard | conversation (ShareGPT JSONL for fine-tuning)
rendering: {image_size: 512, point_radius: 6.0, show_grid: true}  # also colors, marker/marker_a/marker_b
```

Transform names per coordinate system (others are silently ignored):

| coordinate_system | transforms | direction vocabulary |
|---|---|---|
| `linear_1d` | `translation_x` | left, right |
| `cartesian_2d` | `translation_x`, `translation_y`, `rotation_deg`, `scale` | left, right, up, down, clockwise, counterclockwise |
| `polar` | `radial_shift`, `angular_rotation_deg`, `radial_scale` | inward, outward, clockwise, counterclockwise |

`points.aspect_ratio` (≠ 1.0) is valid only for `cartesian_2d`.

## Code map

- `sophon/schema.py`: Pydantic models (`BenchmarkSpec`, `SuiteSpec`, `Sample`)
- `sophon/coordinates/`: one class per coordinate system (`base.py` ABC, `registry.py` factory)
- `sophon/transforms.py`: transform sampling
- `sophon/renderer.py`: matplotlib → PIL rendering
- `sophon/questions.py`: question text + ground truth per type
- `sophon/generator.py`: spec + seed → sample iterator (incl. `balance: exact` scheduling)
- `sophon/parsing.py`: free-form VLM text → structured answer
- `sophon/evaluator.py`: OpenAI-compatible adapter, scoring, results aggregation
- `sophon/cli.py`: `generate`, `preview`, `evaluate`, `score`

To add a coordinate system, subclass `coordinates/base.py`, register it in
`registry.py`, and add the literal to `schema.py`.

## Invariants (don't break these)

- **Determinism.** Same spec + seed = byte-identical images and questions. All
  randomness goes through the seeded `numpy.random.Generator`. Never use
  `set`/`dict` iteration order, `hash()`, or global RNG state anywhere that
  affects output (an earlier bug was option order changing with `PYTHONHASHSEED`).
  New spec knobs must default to a no-op so existing specs regenerate identically.
- **Rendering versions are pinned** (numpy/matplotlib/Pillow in `pyproject.toml`)
  because they affect pixels. Don't loosen them casually.
- **Categorical = one transform type per sample.** Compound transforms have no
  single correct direction, so compound specs omit categorical questions.
  Answers are the *correction* (inverse of the applied transform). In Cartesian,
  image y is flipped; "up" means up on screen.
- **No degenerate stimuli.** The generator rejects transforms that do nothing
  visible and questions whose label is constant. Rotation ground truth follows
  the shortest arc.
- **Binary:** aligned = small *coherent* transform (≈0.25–1 dot diameter), never
  per-point jitter; misaligned ≥ ~3–4 dot diameters; no borderline cases.
- **Comparative:** must have a clear winner (aligned vs misaligned, or ≥2–3×
  magnitude ratio).
- **Quantitative:** magnitude only (positive degrees or scale ratio); no
  translation quantities.
- **Parsing:** last relevant keyword wins; quantitative prefers unit-adjacent
  numbers and rejects list markers. `parse_success=False` means "no answer
  attempt", not "wrong". It's tracked separately from accuracy.

## Lint, CI, release

- `uv run ruff format . && uv run ruff check .` (line length 100). CI gates on
  a narrower rule set: `ruff check sophon --select E9,F63,F7,F82,F401,F541,F811`.
- CI (`.github/workflows/ci.yml`): lint, tests on 3.11–3.13, and an
  installed-wheel smoke test.
- Releases are tag-driven via GitHub Releases + PyPI Trusted Publishing. See
  `docs/RELEASING.md`. In short: bump `version` in `pyproject.toml`, merge,
  `make tag`, `gh release create v<version> --generate-notes`.
  `sophon.__version__` comes from package metadata, so there's only one place
  to bump.
