"""Question text and ground truth generation for each question type."""

from __future__ import annotations

from sophon.coordinates.base import CoordinateSystem
from sophon.transforms import invert_direction

# --- Question templates ---

BINARY_TEMPLATE = (
    "Are the blue and red point sets aligned to within approximately one dot width?"
)

CATEGORICAL_TEMPLATE = (
    "Which direction should the red points move over the shortest path "
    "to align with the blue points? Choose from: {directions}"
)

COMPARATIVE_TEMPLATE = (
    "Which image shows better alignment between the blue and red point sets, A or B?"
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
    "Estimate the rotation angle in degrees between the blue and red point sets."
)

_QUANTITATIVE_TEMPLATES: dict[str, str] = {
    # Rotation — all use same template (magnitude only, no direction)
    "rotation_deg": _ROTATION_TEMPLATE,
    "angular_rotation_deg": _ROTATION_TEMPLATE,
    "azimuth_rotation_deg": _ROTATION_TEMPLATE,
    "roll_deg": _ROTATION_TEMPLATE,
    # Scale
    "scale": "By what factor are the red points scaled relative to the blue points?",
    "radial_scale": (
        "By what factor are the red points radially scaled"
        " relative to the blue points?"
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
) -> tuple[str, str]:
    """Return (question_text, ground_truth: direction word).

    The answer is the *inverse* of the applied direction: we shifted right,
    so the correction direction is left.
    """
    directions = ", ".join(coord_system.direction_vocabulary)
    question = CATEGORICAL_TEMPLATE.format(directions=directions)
    applied_direction = coord_system.classify_direction(transform_params)
    answer = invert_direction(applied_direction)
    return question, answer


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
) -> tuple[str, float] | None:
    """Return (question_text, ground_truth) or None if no rotation/scale active.

    Asks for magnitude only, not direction. This avoids sign ambiguity
    (e.g., -34° vs +326°) and cleanly separates magnitude estimation
    (tested here) from direction judgment (tested by categorical).

    Ground truth:
    - Rotation: positive magnitude (absolute value of applied rotation)
    - Scale: actual scale factor (1 + scale_param)
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
        # Scale: actual factor applied
        ground_truth = 1.0 + transform_params.get(best_key, 0.0)
    else:
        # Rotation: positive magnitude (no direction)
        ground_truth = abs(transform_params.get(best_key, 0.0))

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
