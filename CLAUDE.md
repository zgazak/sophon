# CLAUDE.md — Sophon

## What Is This

Sophon is a **generative benchmark for probing spatial reasoning in VLMs across coordinate geometries**. It is named after the Sophon probes from Cixin Liu's *Three-Body Problem* — entities that operate across dimensional spaces to observe and probe the limits of understanding.

The core scientific question: **can Vision-Language Models understand and reason within coordinate systems of increasing geometric complexity?**

Existing spatial reasoning benchmarks (What's Up, CLEVR, SpartUN) test discrete spatial relations between objects ("the red ball is left of the blue cube"). Sophon tests something fundamentally different: whether VLMs understand the **geometry of the space itself**. Two sets of colored points exist in a shared coordinate system, and the model must reason about their geometric relationship — translation, rotation, scale — within that coordinate system. As the geometry grows more complex, the same transformation types become harder to reason about.

## Package Identity

- **PyPI name:** `sophon-bench`
- **Import name:** `sophon`
- **CLI command:** `sophon`
- **License:** MIT
- **Python:** >= 3.11
- **Package manager:** uv
- **Dependencies (core):** numpy, matplotlib, Pillow, pydantic (v2), PyYAML, click
- **Dependencies (eval):** transformers, torch, openai, anthropic, accelerate — optional `[eval]` extra
- **Dependencies (dev):** pytest, pytest-cov, ruff — optional `[dev]` extra
- **Build backend:** hatchling

## Core Design: Generator-First

**Sophon ships NO images.** It ships a generator and lightweight dataset specs (YAML files, ~1KB each).

```
spec (YAML) + seed (int) → deterministic (image, question, ground_truth) triples
```

This makes the benchmark:
- **Contamination-resistant** — regenerate with a new seed anytime
- **Infinitely extensible** — a new YAML spec = a new dataset
- **Tiny to distribute** — pip install + YAML files, no multi-GB downloads
- **Usable online or batch** — generate per-sample during VLM eval, or batch-generate to disk for training

All randomness flows through `numpy.random.Generator(PCG64(seed))`. Same spec + same seed = identical output on any machine. This is non-negotiable.

## The Visual Task

Two sets of colored dots are rendered in a shared coordinate system:
- **Blue dots** (#2563EB): the reference set
- **Red dots** (#DC2626): a transformed copy of the reference set

The VLM is shown the image and asked to reason about the geometric relationship between the two sets.

### Visual Design Decisions
- White/light gray background (#F8F8F8)
- All dots are the **same size** within an image. No size-based encoding.
- Simple filled circles, no borders, slight alpha (0.8) so overlaps are visible
- Optional coordinate-system-appropriate grid lines (Cartesian grid, polar concentric rings + radial lines, etc.) to hint at the geometry
- 512×512 default resolution, configurable
- **CPU-only rendering** with matplotlib. No GPU needed for generation. This must be fast — thousands of images per minute on a laptop.
- Blue and red chosen for colorblind safety (the problematic axis is red/green, not red/blue)
- Two colors only. The task is geometric reasoning, not object tracking.
- For comparative questions: side-by-side A|B layout in a single image with labels

## Coordinate System Progression

Four coordinate systems of increasing geometric complexity. Each level contains the previous as a special case — Cartesian is polar with infinite radius, and models that understand deeper geometry should handle simpler cases automatically.

### Level 1: 1D Linear
- Points on a number line segment
- **Transforms:** translation only
- **Rendering:** horizontal line with tick marks, dots plotted along it (blue above line, red below, or interleaved with color distinction)
- **Direction vocabulary:** left, right
- This is the absolute floor. If a VLM can't do this, that's a finding.

### Level 2: 2D Cartesian
- Points on a [0,1]×[0,1] grid
- **Transforms:** translation (x, y), rotation (degrees, around centroid), scale (from centroid)
- **Rendering:** optional grid lines, y-axis flipped for image convention
- **Direction vocabulary:** left, right, up, down, clockwise, counterclockwise
- The workhorse level — most existing spatial reasoning intuition lives here

### Level 3: Polar
- Points in (r, θ) space, r ∈ [0,1], θ ∈ [0, 2π)
- **Transforms:** radial shift, angular rotation, radial scale
- **Rendering:** concentric circle grid, radial line grid, origin at image center
- **Direction vocabulary:** inward, outward, clockwise, counterclockwise
- **Key test:** "shift right" no longer means the same thing everywhere in the image. A model that learned Cartesian heuristics ("dots on the right side of the image = shifted right") will fail here.
- **Point generation:** use r ~ sqrt(U) for area-uniform density on the disk

### Level 4: Spherical (Fisheye)
- Points on a hemisphere in (altitude, azimuth), projected to a disk via equidistant fisheye projection: r = f × zenith_angle
- altitude: 0 = horizon, π/2 = zenith. azimuth: 0 = north, increasing clockwise
- **Transforms:** azimuth rotation, altitude offset, roll
- **Rendering:** circular disk, concentric altitude rings, radial azimuth lines. Points near center (zenith) behave almost Cartesian; points near edge (horizon) are heavily distorted.
- **Direction vocabulary:** rotate_azimuth_cw, rotate_azimuth_ccw, shift_zenith_in, shift_zenith_out, roll_cw, roll_ccw
- **Self-contained projection** — no external dependencies. The equidistant fisheye formula is trivial: r = f × (π/2 - altitude).
- This is the hardest level. It's motivated by real all-sky camera alignment problems (see the `allclear` project), but the benchmark uses abstract dots, not astronomical imagery.

**NOTE:** Projective (3D perspective) was considered and deferred to v0.2. Depth encoding with uniform-size dots is visually ambiguous — there's no clean way to convey depth without introducing point-size conventions that become a separate thing being tested.

## Question Taxonomy

For each coordinate system, four question types of increasing difficulty. This creates a 2D evaluation matrix: coordinate complexity × question difficulty.

### Binary
- "Are the blue and red point sets aligned to within approximately one dot width?" → yes / no
- **Must generate balanced examples** — ~50% aligned (offset ≤ ~1 dot diameter), ~50% misaligned (offset ≥ ~3-4 dot diameters). Avoid borderline cases in between.
- The "one dot width" framing gives the model a concrete visual reference for the threshold. The dots are in the image; their size is visible; "within one dot width" is judgeable by looking.
- **Aligned samples use a small coherent transform** (minimum ~0.25 dot diameters, maximum ~1 dot diameter). This keeps both colors visible (no perfect overlap) while preserving the identical spatial pattern between sets. Do NOT use per-point random jitter — jitter changes the pattern itself, and a model could legitimately say "these are different arrangements that happen to be close" which is not the same as "aligned."
- **Misaligned samples must be clearly misaligned** — minimum transform magnitude should produce at least 3-4 dot diameters of offset so there's no ambiguity.

### Categorical
- "Which direction should the red points move to align with the blue points?" → one of the coordinate system's direction vocabulary
- The answer is the **inverse** of the applied transformation direction (we applied "shift right", so the correction is "left")
- Transformation must be large enough to be unambiguous (minimum magnitude in spec)
- **Categorical specs must use single-transform-type only.** Each sample applies exactly ONE type of transform — either translation_x OR translation_y OR rotation OR scale, never combinations. If a compound transform is applied (e.g., shift right AND down), the "correct" single-direction answer is ambiguous and the question is unfair. Compound transforms are tested via binary and comparative questions where you don't need to name a single direction.
- This means categorical specs should isolate individual axes: `cartesian_translation_x.yaml`, `cartesian_translation_y.yaml`, `cartesian_rotation.yaml`, etc. A `cartesian_compound.yaml` spec should include binary, comparative, and quantitative but NOT categorical.

### Comparative
- Show two images side-by-side (A and B), same reference points but different transformations
- "Which image shows better alignment between the blue and red point sets, A or B?" → A or B
- Better alignment = smaller transformation magnitude
- Rendered as a single wide image with A|B labels
- **The comparison must have a clear winner.** If both panels are badly misaligned, the question is a meaningless judgment call. Constrain the two transform magnitudes so the "better" one is obviously better:
  - **Easy tier:** One panel is "aligned" (per-point jitter only, same as binary aligned samples) and the other is clearly misaligned (≥3-4 dot diameters offset). This is almost a binary question in comparative format.
  - **Hard tier:** Both panels are misaligned, but one has at least 2-3x larger transform magnitude than the other. E.g., 5° rotation vs 20° rotation — not 15° vs 20°.
  - **Never generate:** Two panels where both magnitudes are large and within ~1.5x of each other. These are coin flips, not spatial reasoning tests.

### Quantitative
- **Magnitude only, no direction.** Quantitative questions ask "how much" not "which way" — direction is already tested by categorical questions. This avoids sign ambiguity (is -34° or +326° the "correct" correction?) and keeps scoring clean.
- **Rotation:** "Estimate the rotation angle in degrees between the blue and red point sets." Answer is always a positive magnitude (e.g., 34.6, not -34.6). Degrees are universally understood, require no coordinate labels, and work across all coordinate systems.
- **Scale:** "By what factor are the red points scaled relative to the blue points?" Answer is the observed ratio (e.g., 2.0 if red dots are spread twice as far from center). Not "what factor to correct" — just "what is the relationship."
- **No quantitative translation.** Translation magnitudes require labeled axes or arbitrary coordinate conventions — the model would be guessing at units, not reasoning spatially.
- Free-form numeric answer. Scored by absolute error and binned accuracy (e.g., within 5°, 10°, 25° for rotation; within 0.1x, 0.25x for scale).

### Coordinate System Identification (standalone task)
- Show a single set of points **without grid lines** — the model should identify the coordinate system from the point distribution alone (uniform density = Cartesian, 1/r radial density = circular, collinear = linear)
- Use a single neutral color for points (e.g., purple #7C3AED) since there's only one set, not the blue+red pair
- **3-way classification**, not 4-way: "What type of coordinate system are these points arranged in? Choose from: 1D linear, 2D Cartesian, circular (polar/spherical)."
- Polar and spherical are visually indistinguishable from point distributions alone (both produce 1/r radial density falloff via identical Jacobians). Grouping them as "circular" is honest. The polar vs spherical distinction is tested by the transform questions, where their behavior diverges.
- Tests prerequisite knowledge — if a VLM can't identify the coordinate system, there's no point testing whether it can reason about transformations within one

### Grid Visibility Rule
- **Grids OFF for coord_id questions.** The model should recognize the coordinate system from point distributions, not grid line patterns.
- **Grids ON for all transform questions** (binary, categorical, comparative, quantitative). Grid lines provide visual anchors for directional vocabulary ("inward", "clockwise", etc.).
- This means grid visibility is per-question-type, not just per-spec. The renderer should accept a `show_grid` override, and the generator should force `show_grid=False` for coord_id samples regardless of spec settings.

## Spec Schema

Each dataset is defined by a YAML spec. Here's the structure (use Pydantic v2 for validation):

```yaml
name: cartesian_rotation           # Unique identifier
description: "..."                 # Human-readable description
coordinate_system: cartesian_2d    # One of: linear_1d, cartesian_2d, polar, spherical

rendering:
  image_size: 512
  background_color: "#F8F8F8"
  color_a: "#2563EB"               # Blue (reference)
  color_b: "#DC2626"               # Red (transformed)
  point_radius: 6.0
  show_grid: true
  grid_color: "#E0E0E0"
  grid_alpha: 0.5

points:
  num_points: [30, 60]             # Uniform random int in this range per sample
  distribution: uniform            # uniform | clustered | grid
  margin: 0.1                      # Fraction of space left empty at edges

transforms:                        # Each is a {min, max} range. Omit or set to [0,0] to disable.
  rotation_deg: [-30, 30]          # 2D Cartesian rotation
  translation_x: [-0.15, 0.15]    # Cartesian / 1D translation
  # ... coordinate-system-specific fields ...

questions:
  - type: binary
    alignment_threshold: 0.02
  - type: categorical
  - type: comparative
  - type: quantitative

num_samples: 1000
seed: 42
```

Transform field names should be consistent with the coordinate system's supported_transforms. The generator silently ignores transform fields that don't apply to the active coordinate system.

## Package Structure

```
sophon-bench/
├── pyproject.toml
├── README.md
├── CLAUDE.md                        # This file
├── LICENSE
├── sophon/
│   ├── __init__.py                  # Version, top-level imports
│   ├── schema.py                    # Pydantic v2 models: BenchmarkSpec, Sample, etc.
│   ├── coordinates/
│   │   ├── __init__.py              # CoordinateSystem ABC
│   │   ├── registry.py              # Type → class mapping, factory function
│   │   ├── linear.py                # 1D Linear
│   │   ├── cartesian.py             # 2D Cartesian
│   │   ├── polar.py                 # Polar
│   │   └── spherical.py             # Spherical / Fisheye
│   ├── transforms.py                # Transform sampling from ranges
│   ├── renderer.py                  # matplotlib rendering → PIL Image
│   ├── questions.py                 # Question text + ground truth generation
│   ├── generator.py                 # Spec + seed → (image, question, answer) iterator
│   ├── parsing.py                   # Response parser: extract answers from free-form VLM text
│   ├── evaluator.py                 # Score VLM responses against ground truth
│   ├── cli.py                       # Click CLI: generate, preview, evaluate, score
│   └── specs/                       # Canonical benchmark specs shipped with package
│       ├── linear_translation.yaml
│       ├── cartesian_translation.yaml
│       ├── cartesian_rotation.yaml
│       ├── cartesian_compound.yaml
│       ├── polar_rotation.yaml
│       ├── polar_radial.yaml
│       ├── spherical_azimuth.yaml
│       ├── spherical_altitude.yaml
│       ├── coord_identification.yaml
│       └── full_suite.yaml          # Meta-spec referencing all canonical specs
├── tests/
│   ├── test_coordinates.py          # Math correctness for each coordinate system
│   ├── test_transforms.py           # Transform application + inversion
│   ├── test_renderer.py             # Rendering determinism
│   ├── test_generator.py            # End-to-end spec → samples
│   ├── test_questions.py            # Question/answer generation
│   ├── test_parsing.py              # Response parser (free-form VLM text → structured answers)
│   └── test_schema.py               # Spec validation
└── scripts/
    ├── evaluate_vlm.py              # Run a VLM against Sophon specs
    ├── generate_batch.py            # Batch generate images to disk
    └── visualize_results.py         # Plot results heatmaps
```

## Coordinate System ABC

Each coordinate system class implements:

1. **`generate_points(n, rng, distribution, margin) → ndarray(n, ndim)`** — Points in native coordinate space
2. **`apply_transform(points, transform_params) → ndarray`** — Transform in native space
3. **`to_image_coords(points, image_size, margin) → ndarray(n, 2)`** — Native → pixel projection
4. **`render_grid(ax, image_size, margin, color, alpha)`** — Draw coordinate grid on matplotlib Axes
5. **`describe_transform(transform_params) → str`** — Human-readable description for answer generation
6. **`classify_direction(transform_params) → str`** — Map transform to one of `direction_vocabulary`
7. **`transform_magnitude(transform_params) → float`** — Scalar magnitude for thresholding/comparison
8. **`sample_transform(rng, transform_ranges) → dict`** — Sample random transform from spec ranges

Properties: `name`, `ndim`, `direction_vocabulary`, `supported_transforms`

Direction answers for categorical questions use the **inverse** of the applied direction ("we shifted right, so the correction is left"). The questions module handles inversion.

## CLI Commands

```bash
# Generate a full dataset to disk
sophon generate --spec specs/cartesian_rotation.yaml --seed 42 --output-dir output/

# Quick visual inspection
sophon preview --spec specs/polar_rotation.yaml --num-samples 5 --output-dir preview/

# Evaluate a VLM via OpenAI-compatible API (vllm, ollama, etc.)
sophon evaluate --spec specs/full_suite.yaml --seed 42 \
    --backend openai --model-name llava-v1.6-7b --api-base http://localhost:8000/v1

# Evaluate a local transformers model
sophon evaluate --spec specs/full_suite.yaml --seed 42 \
    --backend transformers --model-name llava-hf/llava-v1.6-mistral-7b-hf

# Score existing predictions file
sophon score --predictions results/predictions.json --ground-truth results/ground_truth.json
```

## Evaluation & Scoring

The evaluator takes a callable `vlm_fn(image: PIL.Image, question: str) → str` and scores against ground truth.

### Response Parsing

VLMs return free-form text, not structured answers. The `parsing.py` module is responsible for extracting the actual answer from whatever the model says. This is critical — a fragile parser will silently corrupt results. It should be its own module, not buried inside the evaluator.

#### Format Suffixes

Every question type appends a short format instruction to the question
text. This is standard practice in VLM benchmarks (MMMU, MathVista,
BLINK). The suffix is directive but not so tight that every response
becomes trivially parseable — models that can't engage with the task
should still produce unparseable responses.

| Type | Suffix |
|------|--------|
| Binary | "Answer yes or no." |
| Categorical | "Answer with one of: {direction vocabulary}." |
| Comparative | "Answer A or B." |
| Quantitative (rotation) | "State your estimate in degrees as a number." |
| Quantitative (scale) | "State your estimate as a numeric factor." |
| Coord identification | Already has "Choose from: ..." (no change) |

The suffix is part of the question text in the user message, not a
system prompt. The categorical suffix dynamically includes the
coordinate system's direction vocabulary.

#### Last-Keyword-Wins Rule

**All parsers prefer the LAST relevant keyword, not the first.** Models
often restate the problem before answering:
- "The question asks if they are aligned. No, they are not aligned."
  First keyword = "aligned" (yes), last keyword = "not" (no, correct).
- "The red points are shifted right. To align, move left."
  First direction = "right", last direction = "left" (correct).

This applies to binary, categorical, comparative, and coord
identification parsers.

#### Quantitative Parser Priority Order

The quantitative parser uses a 3-tier extraction strategy to avoid
picking up list markers ("1. Normalize", "2. Calculate") from tutorial
responses:

1. **Unit-adjacent numbers** — "30 degrees", "factor of 2.3", "30°"
2. **Last non-list-marker number** — filters out numbers followed by
   `. [A-Z]`, `. **`, `)`, or preceded by "Step"/"Item"
3. **Reject** — if only list markers found, `parse_success=False`

#### parse_success Semantics

`parse_success=False` means "no genuine answer attempt detected", NOT
"model got it wrong." A model that writes a tutorial about how one
*could* estimate rotation (without actually doing it) gets
`parse_success=False`. A model that says "25 degrees" when the answer
is 30 gets `parse_success=True, correct=False`.

Parsing rules per question type:
- **Binary:** Search for all occurrences of yes/no keywords. Take the last one (last-keyword-wins). If neither found, mark unparseable.
- **Categorical:** Search for all direction vocabulary words. Take the last one found. If none found, mark unparseable.
- **Comparative:** Search for all A/B occurrences. Take the last one. If neither found, mark unparseable.
- **Quantitative:** Priority extraction as described above. Rejects list markers and tutorial-only responses.
- **Coord identification:** Search for coordinate system aliases. Take the last match found.

Unparseable responses should be tracked separately in the results — they indicate prompt/format issues or model inability to engage, not spatial reasoning failures. Report parse rate alongside accuracy.

### Scoring

**Scoring per question type:**
- Binary: accuracy (with chance baseline = 50%)
- Categorical: accuracy (with chance baseline = 1/len(direction_vocabulary) — varies by coordinate system)
- Comparative: accuracy (with chance baseline = 50%)
- Quantitative: mean absolute error + binned accuracy at configurable thresholds (e.g., within 5°, 10°, 25°)
- Coord identification: accuracy (with chance baseline = 25% for 4 coordinate systems)

Always report chance baselines alongside model performance so readers can immediately calibrate.

### Output Format

The evaluator produces two output files:

1. **Per-sample predictions** (`predictions.jsonl`): One JSON object per line with `sample_id`, `question_type`, `coordinate_system`, `ground_truth`, `raw_response`, `parsed_answer`, `correct`, `error` (for quantitative). This is the raw data — it enables post-hoc analysis of which specific transformations, magnitudes, or directions a model gets right/wrong without re-running evaluation.

2. **Aggregated results** (`results.json`): Per-coordinate-system × per-question-type scores, parse rates, and chance baselines. This is what gets reported in papers and plotted as the heatmap.

The key visualization is a **heatmap** with coordinate systems on one axis and question types on the other — this is the main figure in the paper.

### Suite Versioning

Canonical benchmark specs have a version string. `sophon evaluate --suite v0.1` always generates the exact same samples (same specs, same seed). If specs change between releases, the version increments. This ensures results are comparable across papers and time. The `full_suite.yaml` meta-spec includes a `version` field.

### VLM Adapters

Built-in VLM adapters for: OpenAI-compatible API (covers vllm-served models), Anthropic API, local HuggingFace transformers pipeline. Each adapter normalizes to the `vlm_fn` interface. The OpenAI-compatible adapter is the most important for reproducibility — it's how frontier models (GPT-4o, Gemini, Claude) are accessible, and it's how most people serve local models via vllm/ollama. It should handle rate limiting and retries gracefully.

## Development Workflow

```bash
uv sync                    # Install with dev deps
uv run pytest              # Run tests
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv run sophon preview --spec sophon/specs/cartesian_rotation.yaml --num-samples 3
```

Use `ruff` for formatting and linting. Type hints everywhere. Test everything — coordinate math, rendering determinism (same spec+seed = same pixel output), question generation, scoring logic.

## Key Principles

1. **Deterministic.** Same spec + same seed = byte-identical output. All randomness from seeded numpy Generator.
2. **Minimal.** CPU rendering, simple visuals, tiny specs. Core package has no torch/transformers dependency.
3. **Extensible.** New coordinate system = one new class. New question type = one new function. New dataset = one YAML file.
4. **Honest.** This is a diagnostic tool, not a leaderboard. The goal is understanding where spatial reasoning breaks down, not ranking models.
5. **The benchmark IS the generator.** Never ship pre-generated images as the primary artifact.

## Scientific Context

This project grew out of a concrete failure: when using VLMs (including Claude) as coding assistants for astronomical instrument calibration (`allclear` project — all-sky camera cloud detection via stellar transmission mapping), the VLM could not reason about how to adjust a geometric model overlay to better match real star positions. It couldn't understand "rotate this fit 20 degrees and see how it improves."

A prior project (ORIENT) investigated injecting Gabor wavelet features into CLIP ViT to improve spatial representations. The internal representations improved dramatically (linear probes confirmed it), but this did NOT translate to downstream VLM improvements — the decoder was never trained to read the enriched features. **The lesson: encoder-level interventions are insufficient without end-to-end training.**

Sophon takes a different approach: instead of proposing a fix, **diagnose where the failure actually is.** The benchmark creates a controlled environment where we can measure exactly what VLMs can and can't do with coordinate geometry, across a clean difficulty progression. The findings should inform where future interventions should target.

## What This Is NOT

- Not a training dataset (though it can be used as one)
- Not proposing a new model or architecture
- Not shipping pre-generated images
- Not specific to astronomy (that's the motivation, but the benchmark is abstract)
- Not a leaderboard — it's a diagnostic tool