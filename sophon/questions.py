"""Question text and ground truth generation for each question type."""

from __future__ import annotations

from sophon.coordinates.base import CoordinateSystem
from sophon.transforms import invert_direction

# ---------------------------------------------------------------------------
# Restricted vocabulary: maps (coordinate_system, transform_type) to the
# minimal direction pair relevant to that transform.  This prevents VLMs
# from defaulting to rotation vocabulary when irrelevant rotation options
# are present (empirically observed: GPT-4o picks "clockwise" for pure
# horizontal translations when full 6-direction vocab is offered).
# ---------------------------------------------------------------------------

RESTRICTED_VOCAB: dict[tuple[str, str], list[str]] = {
    # Cartesian 2D — scale is excluded because classify_direction does not
    # map scale to a direction in this coordinate system.
    ("cartesian_2d", "translation_x"): ["left", "right"],
    ("cartesian_2d", "translation_y"): ["up", "down"],
    ("cartesian_2d", "rotation_deg"): ["clockwise", "counterclockwise"],
    # Linear 1D
    ("linear_1d", "translation_x"): ["left", "right"],
    # Polar
    ("polar", "angular_rotation_deg"): ["clockwise", "counterclockwise"],
    ("polar", "radial_shift"): ["inward", "outward"],
    ("polar", "radial_scale"): ["inward", "outward"],
    # Spherical — uses system-specific direction vocabulary
    ("spherical", "azimuth_rotation_deg"): ["rotate_azimuth_cw", "rotate_azimuth_ccw"],
    ("spherical", "altitude_offset_deg"): ["shift_zenith_in", "shift_zenith_out"],
    ("spherical", "roll_deg"): ["roll_cw", "roll_ccw"],
}


def get_restricted_vocab(
    coord_system_name: str, transform_type: str
) -> list[str] | None:
    """Return the restricted direction vocabulary for a (coord_system, transform) pair.

    Returns None if no restricted vocabulary is defined (falls back to full).
    """
    return RESTRICTED_VOCAB.get((coord_system_name, transform_type))


def detect_active_transform(transform_params: dict[str, float]) -> str | None:
    """Return the single active (nonzero) transform key, or None if ambiguous."""
    active = [k for k, v in transform_params.items() if abs(v) > 1e-9]
    if len(active) == 1:
        return active[0]
    return None

# --- Question templates ---
# Each template ends with a format suffix that tells the model what
# form the answer should take. This is standard practice in VLM
# benchmarks (MMMU, MathVista, BLINK). The suffix is directive but
# not so tight that every response becomes trivially parseable —
# models that can't engage with the task should still produce
# unparseable responses.

BINARY_TEMPLATE = (
    "Are the blue and red point sets aligned to within approximately one dot width? "
    "Answer yes or no."
)

_CATEGORICAL_TEMPLATE_DEFAULT = (
    "Which direction should the red points move "
    "to align with the blue points? Answer with one of: {directions}."
)

_CATEGORICAL_TEMPLATE_ROTATION = (
    "Which direction should the red points move over the shortest arc "
    "to align with the blue points? Answer with one of: {directions}."
)

# Transform types where rotation/arc ambiguity applies
_ANGULAR_TRANSFORM_TYPES = {
    "rotation_deg",
    "angular_rotation_deg",
    "azimuth_rotation_deg",
    "roll_deg",
}

COMPARATIVE_TEMPLATE = (
    "Which image shows better alignment between the blue and red point sets? "
    "Answer A or B."
)

COORD_ID_TEMPLATE = (
    "These points are uniformly distributed in their native coordinate space. "
    "What type of coordinate system are they arranged in? "
    "Choose from: 1D linear, 2D Cartesian, circular"
)

# Quantitative: rotation and scale only (not translation — no axis labels to judge units)
_QUANTITATIVE_ROTATION_KEYS = {
    "rotation_deg",
    "angular_rotation_deg",
    "azimuth_rotation_deg",
    "roll_deg",
}

_QUANTITATIVE_SCALE_KEYS = {
    "scale",
    "radial_scale",
}

QUANTITATIVE_KEYS = _QUANTITATIVE_ROTATION_KEYS | _QUANTITATIVE_SCALE_KEYS

_ROTATION_TEMPLATE = (
    "Estimate the rotation angle in degrees between the blue and red point sets. "
    "State your estimate in degrees as a number."
)

_QUANTITATIVE_TEMPLATES: dict[str, str] = {
    # Rotation — all use same template (magnitude only, no direction)
    "rotation_deg": _ROTATION_TEMPLATE,
    "angular_rotation_deg": _ROTATION_TEMPLATE,
    "azimuth_rotation_deg": _ROTATION_TEMPLATE,
    "roll_deg": _ROTATION_TEMPLATE,
    # Scale
    "scale": (
        "By what factor are the red points scaled relative to the blue points? "
        "State your estimate as a numeric factor."
    ),
    "radial_scale": (
        "By what factor are the red points radially scaled relative to the blue points? "
        "State your estimate as a numeric factor."
    ),
}


def has_quantitative_transform(transform_params: dict[str, float]) -> bool:
    """Return True if the transform includes rotation or scale (not just translation)."""
    for key in QUANTITATIVE_KEYS:
        if abs(transform_params.get(key, 0.0)) > 1e-6:
            return True
    return False


def generate_binary_question(
    is_aligned: bool,
) -> tuple[str, str]:
    """Return (question_text, ground_truth: 'yes' | 'no').

    The is_aligned flag is determined by the generator (coin flip),
    not by magnitude thresholding. Aligned samples get per-point jitter
    (visually close pairs), misaligned samples get real transforms
    (clearly separated clouds). No borderline cases.
    """
    answer = "yes" if is_aligned else "no"
    return BINARY_TEMPLATE, answer


def generate_categorical_question(
    coord_system: CoordinateSystem,
    transform_params: dict[str, float],
    vocab_mode: str = "restricted",
    vocab_order: str = "fixed",
    order_seed: int = 0,
) -> tuple[str, str, list[str]]:
    """Return (question_text, ground_truth, valid_directions).

    The answer is the *inverse* of the applied direction: we shifted right,
    so the correction direction is left.

    vocab_mode:
        "restricted" (default) — use only the direction pair relevant to the
        active transform type.  Prevents distractor confusion.
        "full" — use the coordinate system's full direction_vocabulary.

    vocab_order:
        "fixed" (default) — options in canonical order.
        "random" — deterministic random shuffle using *order_seed*.
        "random_correct_last" — correct answer placed last, rest shuffled.
    """
    import random as _random

    applied_direction = coord_system.classify_direction(transform_params)
    answer = invert_direction(applied_direction)

    # Determine vocabulary
    if vocab_mode == "restricted":
        transform_type = detect_active_transform(transform_params)
        vocab = None
        if transform_type is not None:
            vocab = get_restricted_vocab(coord_system.name, transform_type)
        if vocab is None:
            vocab = list(coord_system.direction_vocabulary)
    else:
        vocab = list(coord_system.direction_vocabulary)

    assert answer in vocab, (
        f"GT '{answer}' not in vocab {vocab} "
        f"for ({coord_system.name}, {detect_active_transform(transform_params)})"
    )

    # Apply vocab ordering
    if vocab_order == "random":
        rng = _random.Random(order_seed)
        vocab = list(vocab)
        rng.shuffle(vocab)
    elif vocab_order == "random_correct_last":
        rng = _random.Random(order_seed)
        others = [v for v in vocab if v != answer]
        rng.shuffle(others)
        vocab = others + [answer]

    # Pick template: "shortest arc" only for rotation/angular transforms
    transform_type = detect_active_transform(transform_params)
    if transform_type is not None and transform_type in _ANGULAR_TRANSFORM_TYPES:
        template = _CATEGORICAL_TEMPLATE_ROTATION
    else:
        template = _CATEGORICAL_TEMPLATE_DEFAULT

    directions = ", ".join(vocab)
    question = template.format(directions=directions)
    return question, answer, vocab


def generate_comparative_question(
    coord_system: CoordinateSystem,
    transform_params_a: dict[str, float],
    transform_params_b: dict[str, float],
    label_better: str,
) -> tuple[str, str]:
    """Return (question_text, ground_truth: 'A' | 'B')."""
    return COMPARATIVE_TEMPLATE, label_better


def generate_quantitative_question(
    coord_system: CoordinateSystem,
    transform_params: dict[str, float],
    range_hint: bool = False,
    transform_ranges: dict[str, list[float]] | None = None,
) -> tuple[str, float] | None:
    """Return (question_text, ground_truth) or None if no rotation/scale active.

    Asks for magnitude only, not direction. This avoids sign ambiguity
    (e.g., -34° vs +326°) and cleanly separates magnitude estimation
    (tested here) from direction judgment (tested by categorical).

    Ground truth:
    - Rotation: positive magnitude (absolute value of applied rotation)
    - Scale: actual scale factor (1 + scale_param)

    range_hint:
        When True and *transform_ranges* is provided, appends the valid
        magnitude range to the question prompt.  This is a prompt-only
        ablation — same images, same ground truth, just extra info.
    """
    best_key = None
    best_mag = -1.0
    for key in QUANTITATIVE_KEYS:
        val = abs(transform_params.get(key, 0.0))
        if val > best_mag:
            best_mag = val
            best_key = key

    if best_key is None or best_mag < 1e-6:
        return None

    question = _QUANTITATIVE_TEMPLATES[best_key]

    if best_key in _QUANTITATIVE_SCALE_KEYS:
        ground_truth = 1.0 + transform_params.get(best_key, 0.0)
    else:
        ground_truth = abs(transform_params.get(best_key, 0.0))

    # Append range hint if requested
    if range_hint and transform_ranges and best_key in transform_ranges:
        lo, hi = transform_ranges[best_key]
        if best_key in _QUANTITATIVE_SCALE_KEYS:
            # Scale GT is 1 + param, so range is [1+lo, 1+hi]
            question += f" The scale factor is between {1.0 + lo:.2f} and {1.0 + hi:.2f}."
        else:
            # Rotation GT is abs(param), so range is [0, max(|lo|, |hi|)]
            max_deg = max(abs(lo), abs(hi))
            question += f" The rotation angle is between 0 and {max_deg:.0f} degrees."

    return question, ground_truth


def generate_coord_identification_question(
    coord_system: CoordinateSystem,
) -> tuple[str, str]:
    """Return (question_text, ground_truth: coordinate system name).

    3-way classification: linear, Cartesian, circular.
    """
    name_map = {
        "linear_1d": "1D linear",
        "cartesian_2d": "2D Cartesian",
        "polar": "circular",
        "spherical": "circular",  # kept for forward compat
    }
    answer = name_map.get(coord_system.name, coord_system.name)
    return COORD_ID_TEMPLATE, answer
