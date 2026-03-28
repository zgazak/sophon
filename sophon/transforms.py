"""Shared transform utilities: direction inversion, identity transforms."""

from __future__ import annotations

_DIRECTION_INVERSES: dict[str, str] = {
    # Linear / Cartesian
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
    # Rotation
    "clockwise": "counterclockwise",
    "counterclockwise": "clockwise",
    # Polar
    "inward": "outward",
    "outward": "inward",
    # Spherical
    "rotate_azimuth_cw": "rotate_azimuth_ccw",
    "rotate_azimuth_ccw": "rotate_azimuth_cw",
    "shift_zenith_in": "shift_zenith_out",
    "shift_zenith_out": "shift_zenith_in",
    "roll_cw": "roll_ccw",
    "roll_ccw": "roll_cw",
}


def invert_direction(direction: str) -> str:
    """Return the inverse direction (e.g. 'left' -> 'right').

    Used for categorical questions: the answer is the inverse of the applied
    transform direction (we shifted right, so the correction is left).
    """
    if direction not in _DIRECTION_INVERSES:
        raise ValueError(
            f"Unknown direction: {direction!r}. "
            f"Known directions: {sorted(_DIRECTION_INVERSES.keys())}"
        )
    return _DIRECTION_INVERSES[direction]


def make_identity_transform(supported_transforms: list[str]) -> dict[str, float]:
    """Return zero/identity transform params for all supported transform keys."""
    return {key: 0.0 for key in supported_transforms}
