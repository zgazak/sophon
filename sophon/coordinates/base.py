"""Abstract base class for coordinate systems."""

from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.axes
import numpy as np


class CoordinateSystem(ABC):
    """Abstract coordinate system for point generation, transformation, and projection."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this coordinate system (e.g. 'cartesian_2d')."""
        ...

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of native coordinate dimensions."""
        ...

    @property
    @abstractmethod
    def direction_vocabulary(self) -> list[str]:
        """Valid direction words for categorical questions."""
        ...

    @property
    @abstractmethod
    def supported_transforms(self) -> list[str]:
        """Transform parameter names this coordinate system handles."""
        ...

    @abstractmethod
    def generate_points(
        self, n: int, rng: np.random.Generator, distribution: str, margin: float
    ) -> np.ndarray:
        """Generate n points in native coordinate space. Returns ndarray(n, ndim)."""
        ...

    #: Whether a screen-aligned height/width aspect is meaningful in this
    #: system's native coordinates. Polar natives are (r, theta) and linear_1d
    #: has a single axis, so squeezing either would not mean what it says.
    supports_anisotropy: bool = False

    def generate_points_shaped(
        self,
        n: int,
        rng: np.random.Generator,
        distribution: str,
        margin: float,
        aspect_ratio: float = 1.0,
    ) -> np.ndarray:
        """generate_points, then optionally squeeze to a target h/w aspect.

        Backwards compatible by construction: at the default aspect_ratio of 1.0
        this returns generate_points untouched, drawing exactly the same numbers
        from the same rng in the same order.
        """
        points = self.generate_points(n, rng, distribution, margin)
        if aspect_ratio == 1.0:
            return points
        if not self.supports_anisotropy:
            raise ValueError(
                f"aspect_ratio != 1.0 is not supported for coordinate system "
                f"{self.name!r}; only systems with a screen-aligned native "
                f"basis can express a height/width ratio."
            )
        return self._apply_aspect(points, aspect_ratio)

    def _apply_aspect(self, points: np.ndarray, aspect_ratio: float) -> np.ndarray:
        """Squeeze toward the region centre so that height/width == aspect_ratio.

        Only ever compresses (the larger axis keeps its scale), so a cloud that
        respected `margin` still does.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_transform(self, points: np.ndarray, transform_params: dict[str, float]) -> np.ndarray:
        """Apply transformation in native space. Returns transformed points."""
        ...

    @abstractmethod
    def to_image_coords(self, points: np.ndarray, image_size: int, margin: float) -> np.ndarray:
        """Project native coordinates to pixel coordinates. Returns ndarray(n, 2)."""
        ...

    @abstractmethod
    def render_grid(
        self,
        ax: matplotlib.axes.Axes,
        image_size: int,
        margin: float,
        color: str,
        alpha: float,
    ) -> None:
        """Draw coordinate-system-appropriate grid lines on a matplotlib Axes."""
        ...

    @abstractmethod
    def describe_transform(self, transform_params: dict[str, float]) -> str:
        """Human-readable description of the transform for question generation."""
        ...

    @abstractmethod
    def classify_direction(self, transform_params: dict[str, float]) -> str:
        """Map transform to one of direction_vocabulary."""
        ...

    @abstractmethod
    def transform_magnitude(self, transform_params: dict[str, float]) -> float:
        """Scalar magnitude for thresholding and comparison."""
        ...

    @abstractmethod
    def sample_transform(
        self, rng: np.random.Generator, transform_ranges: dict[str, list[float]]
    ) -> dict[str, float]:
        """Sample random transform parameters from spec ranges."""
        ...

    @abstractmethod
    def describe_points(self, points: np.ndarray) -> str:
        """Return a human-readable text description of point positions in native coordinate space.

        Points are sorted by their first coordinate for consistency.
        Coordinates are rounded to 3 decimal places.
        """
        ...
