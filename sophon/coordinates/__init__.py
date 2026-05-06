"""Coordinate system abstractions for Sophon."""

import sophon.coordinates.cartesian  # noqa: F401

# Import all coordinate systems to trigger registration
import sophon.coordinates.linear  # noqa: F401
import sophon.coordinates.polar  # noqa: F401
from sophon.coordinates.base import CoordinateSystem
from sophon.coordinates.registry import get_coordinate_system, register

__all__ = ["CoordinateSystem", "get_coordinate_system", "register"]
