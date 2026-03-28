"""Polar coordinate system."""

from __future__ import annotations

import numpy as np

from sophon.coordinates.base import CoordinateSystem
from sophon.coordinates.registry import register


@register("polar")
class Polar(CoordinateSystem):
    """Points in (r, theta) space with radial and angular transforms."""

    @property
    def name(self) -> str:
        return "polar"

    @property
    def ndim(self) -> int:
        return 2

    @property
    def direction_vocabulary(self) -> list[str]:
        return ["inward", "outward", "clockwise", "counterclockwise"]

    @property
    def supported_transforms(self) -> list[str]:
        return ["radial_shift", "angular_rotation_deg", "radial_scale"]

    def generate_points(
        self, n: int, rng: np.random.Generator, distribution: str, margin: float
    ) -> np.ndarray:
        r_max = 1.0 - margin
        if distribution == "uniform":
            # Area-uniform: r ~ sqrt(U) for uniform density on the disk
            r = np.sqrt(rng.uniform(0, r_max**2, size=n))
            theta = rng.uniform(0, 2 * np.pi, size=n)
            return np.column_stack([r, theta])
        elif distribution == "coord_uniform":
            # Coordinate-uniform: raw uniform in (r, θ). Produces 1/r density
            # in image — center-heavy. Used for coord_id to reveal geometry.
            r = rng.uniform(0, r_max, size=n)
            theta = rng.uniform(0, 2 * np.pi, size=n)
            return np.column_stack([r, theta])
        elif distribution == "clustered":
            n_clusters = rng.integers(2, 5)
            center_r = rng.uniform(0.1, r_max * 0.7, size=n_clusters)
            center_theta = rng.uniform(0, 2 * np.pi, size=n_clusters)
            points = []
            for i in range(n):
                ci = i % n_clusters
                r = rng.normal(center_r[ci], 0.08)
                theta = rng.normal(center_theta[ci], 0.3)
                points.append([max(0, min(r, r_max)), theta % (2 * np.pi)])
            return np.array(points)
        elif distribution == "grid":
            n_r = int(np.ceil(np.sqrt(n)))
            n_theta = int(np.ceil(n / n_r))
            rs = np.linspace(0.05, r_max, n_r)
            thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
            grid = np.array([(r, t) for r in rs for t in thetas])[:n]
            jr = rng.uniform(-0.02, 0.02, size=len(grid))
            jt = rng.uniform(-0.05, 0.05, size=len(grid))
            grid[:, 0] = np.clip(grid[:, 0] + jr, 0, r_max)
            grid[:, 1] = (grid[:, 1] + jt) % (2 * np.pi)
            return grid
        else:
            raise ValueError(f"Unknown distribution: {distribution!r}")

    def apply_transform(
        self, points: np.ndarray, transform_params: dict[str, float]
    ) -> np.ndarray:
        pts = points.copy()
        r, theta = pts[:, 0], pts[:, 1]

        # Radial scale
        rs = transform_params.get("radial_scale", 0.0)
        if rs != 0.0:
            r = r * (1.0 + rs)

        # Radial shift
        shift = transform_params.get("radial_shift", 0.0)
        if shift != 0.0:
            r = r + shift

        r = np.clip(r, 0, 1.0)

        # Angular rotation
        rot_deg = transform_params.get("angular_rotation_deg", 0.0)
        if rot_deg != 0.0:
            theta = (theta + np.radians(rot_deg)) % (2 * np.pi)

        return np.column_stack([r, theta])

    def to_image_coords(
        self, points: np.ndarray, image_size: int, margin: float
    ) -> np.ndarray:
        r, theta = points[:, 0], points[:, 1]
        # Convert to Cartesian in [-1, 1] range
        x_cart = r * np.cos(theta)
        y_cart = r * np.sin(theta)
        # Map [-1, 1] to pixel space with margin
        margin_px = margin * image_size
        usable = (image_size - 2 * margin_px) / 2.0
        center = image_size / 2.0
        x_pixel = center + x_cart * usable
        y_pixel = center - y_cart * usable  # Flip y for image convention
        return np.column_stack([x_pixel, y_pixel])

    def render_grid(self, ax, image_size: int, margin: float, color: str, alpha: float) -> None:
        margin_px = margin * image_size
        usable = (image_size - 2 * margin_px) / 2.0
        center = image_size / 2.0
        # Concentric circles
        for r_frac in np.linspace(0.2, 1.0, 5):
            circle = plt_circle(center, r_frac * usable, color, alpha)
            ax.add_patch(circle)
        # Radial lines
        for angle_deg in range(0, 360, 30):
            angle = np.radians(angle_deg)
            x_end = center + usable * np.cos(angle)
            y_end = center - usable * np.sin(angle)
            ax.plot([center, x_end], [center, y_end],
                    color=color, alpha=alpha, linewidth=0.5, zorder=0)

    def describe_transform(self, transform_params: dict[str, float]) -> str:
        parts = []
        rs = transform_params.get("radial_shift", 0.0)
        rot = transform_params.get("angular_rotation_deg", 0.0)
        rsc = transform_params.get("radial_scale", 0.0)
        if abs(rs) > 1e-6:
            parts.append(f"radial shift {rs:.3f}")
        if abs(rot) > 1e-6:
            parts.append(f"angular rotation {rot:.1f}°")
        if abs(rsc) > 1e-6:
            parts.append(f"radial scale {1.0 + rsc:.3f}x")
        return ", ".join(parts) if parts else "identity"

    def classify_direction(self, transform_params: dict[str, float]) -> str:
        rs = transform_params.get("radial_shift", 0.0)
        rot = transform_params.get("angular_rotation_deg", 0.0)
        rsc = transform_params.get("radial_scale", 0.0)

        candidates: list[tuple[float, str]] = []
        if abs(rs) > 1e-6:
            candidates.append((abs(rs), "outward" if rs > 0 else "inward"))
        if abs(rot) > 1e-6:
            # Positive angle = CCW in math, but y-flip makes it CW in image
            candidates.append(
                (abs(rot) / 180.0, "clockwise" if rot > 0 else "counterclockwise")
            )
        if abs(rsc) > 1e-6:
            candidates.append((abs(rsc), "outward" if rsc > 0 else "inward"))

        if not candidates:
            return "outward"  # fallback

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def transform_magnitude(self, transform_params: dict[str, float]) -> float:
        rs = transform_params.get("radial_shift", 0.0)
        rot = transform_params.get("angular_rotation_deg", 0.0)
        rsc = transform_params.get("radial_scale", 0.0)
        return abs(rs) + abs(rot) / 180.0 + abs(rsc)

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


def plt_circle(center: float, radius: float, color: str, alpha: float):
    """Create a matplotlib Circle patch (no fill)."""
    import matplotlib.patches as mpatches

    return mpatches.Circle(
        (center, center), radius, fill=False, edgecolor=color, alpha=alpha,
        linewidth=0.5, zorder=0,
    )
