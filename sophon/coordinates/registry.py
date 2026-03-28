"""Coordinate system registry and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sophon.coordinates.base import CoordinateSystem

_REGISTRY: dict[str, type[CoordinateSystem]] = {}


def register(name: str):
    """Class decorator to register a coordinate system under a string name."""

    def wrapper(cls: type[CoordinateSystem]) -> type[CoordinateSystem]:
        _REGISTRY[name] = cls
        return cls

    return wrapper


def get_coordinate_system(name: str) -> CoordinateSystem:
    """Instantiate a coordinate system by its registered name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown coordinate system: {name!r}. Available: {available}")
    return _REGISTRY[name]()
