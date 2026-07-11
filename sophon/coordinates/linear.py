"""1D Linear coordinate system."""

from __future__ import annotations

import numpy as np

from sophon.coordinates.base import CoordinateSystem
from sophon.coordinates.registry import register


@register("linear_1d")
class Linear1D(CoordinateSystem):
    """Points on a 1D number line with translation only."""

    @property
    def name(self) -> str:
        return "linear_1d"

    @property
    def ndim(self) -> int:
        return 1

    @property
    def direction_vocabulary(self) -> list[str]:
        return ["left", "right"]

    @property
    def supported_transforms(self) -> list[str]:
        return ["translation_x"]

    def generate_points(
        self, n: int, rng: np.random.Generator, distribution: str, margin: float
    ) -> np.ndarray:
        lo, hi = margin, 1.0 - margin
        if distribution in ("uniform", "coord_uniform"):
            return rng.uniform(lo, hi, size=(n, 1))
        elif distribution == "clustered":
            n_clusters = rng.integers(2, 5)
            centers = rng.uniform(lo, hi, size=n_clusters)
            points = []
            for i in range(n):
                c = centers[i % n_clusters]
                spread = (hi - lo) * 0.08
                pt = rng.normal(c, spread)
                points.append(pt)
            pts = np.array(points).reshape(-1, 1)
            return np.clip(pts, lo, hi)
        elif distribution == "grid":
            pts = np.linspace(lo, hi, n).reshape(-1, 1)
            jitter = rng.uniform(-0.01, 0.01, size=(n, 1))
            return np.clip(pts + jitter, lo, hi)
        else:
            raise ValueError(f"Unknown distribution: {distribution!r}")

    def apply_transform(self, points: np.ndarray, transform_params: dict[str, float]) -> np.ndarray:
        tx = transform_params.get("translation_x", 0.0)
        return points + tx

    def to_image_coords(self, points: np.ndarray, image_size: int, margin: float) -> np.ndarray:
        margin_px = margin * image_size
        usable = image_size - 2 * margin_px
        x_pixel = points[:, 0] * usable + margin_px
        # y = center of image (renderer handles vertical offset per color)
        y_pixel = np.full_like(x_pixel, image_size / 2.0)
        return np.column_stack([x_pixel, y_pixel])

    def render_grid(self, ax, image_size: int, margin: float, color: str, alpha: float) -> None:
        margin_px = margin * image_size
        lo = margin_px
        hi = image_size - margin_px
        center_y = image_size / 2.0
        # Horizontal number line
        ax.plot(
            [lo, hi], [center_y, center_y], color=color, alpha=alpha * 1.5, linewidth=1.5, zorder=0
        )
        # Tick marks
        n_ticks = 11
        tick_len = image_size * 0.015
        for x in np.linspace(lo, hi, n_ticks):
            ax.plot(
                [x, x],
                [center_y - tick_len, center_y + tick_len],
                color=color,
                alpha=alpha * 1.5,
                linewidth=1.0,
                zorder=0,
            )

    def describe_transform(self, transform_params: dict[str, float]) -> str:
        tx = transform_params.get("translation_x", 0.0)
        if abs(tx) < 1e-6:
            return "identity"
        return f"translation {tx:.3f}"

    def classify_direction(self, transform_params: dict[str, float]) -> str:
        tx = transform_params.get("translation_x", 0.0)
        if abs(tx) <= 1e-6:
            raise ValueError(
                "classify_direction: translation_x is zero in "
                f"params={transform_params!r}; cannot classify a direction "
                "for an identity transform."
            )
        return "right" if tx > 0 else "left"

    def transform_magnitude(self, transform_params: dict[str, float]) -> float:
        return abs(transform_params.get("translation_x", 0.0))

    def sample_transform(
        self, rng: np.random.Generator, transform_ranges: dict[str, list[float]]
    ) -> dict[str, float]:
        params: dict[str, float] = {}
        for key in self.supported_transforms:
            if key in transform_ranges:
                lo, hi = transform_ranges[key]
                params[key] = float(rng.uniform(lo, hi))
            else:
                params[key] = 0.0
        return params

    def describe_points(self, points: np.ndarray) -> str:
        sorted_pts = points[points[:, 0].argsort()]
        lines = ["Points along a 1D number line from 0.0 to 1.0:"]
        for i, pt in enumerate(sorted_pts):
            lines.append(f"  Point {i + 1}: position {pt[0]:.3f}")
        return "\n".join(lines)
