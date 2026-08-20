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


def wrap_angle_deg(deg: float) -> float:
    """Wrap an angle into (-180, 180] — the shortest arc.

    Categorical rotation questions ask which way a set should move "over the
    shortest arc", so the ground truth must be computed on the wrapped angle. A
    +190 deg rotation is a 170 deg turn the other way; classifying on the raw
    value labels it as the long way round and contradicts the question text.
    Magnitudes need the same treatment, or a 350 deg rotation reports a
    magnitude of 1.94 when it is a 10 deg turn.

    Only |angle| > 180 is affected, so anything generated within the usual
    +-45 deg training range is unchanged.
    """
    wrapped = (float(deg) + 180.0) % 360.0 - 180.0
    # (-180, 180]: the modulo lands exactly-180 on -180; both are equally
    # correct there and the caller treats that case as degenerate anyway.
    return 180.0 if wrapped == -180.0 else wrapped


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
