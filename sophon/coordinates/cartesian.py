"""2D Cartesian coordinate system."""

from __future__ import annotations

import numpy as np

from sophon.coordinates.base import CoordinateSystem
from sophon.coordinates.registry import register
from sophon.transforms import wrap_angle_deg


@register("cartesian_2d")
class Cartesian2D(CoordinateSystem):
    """Points in [0,1] x [0,1] with translation, rotation, and scale transforms."""

    @property
    def name(self) -> str:
        return "cartesian_2d"

    @property
    def ndim(self) -> int:
        return 2

    @property
    def direction_vocabulary(self) -> list[str]:
        return ["left", "right", "up", "down", "clockwise", "counterclockwise"]

    @property
    def supported_transforms(self) -> list[str]:
        return ["translation_x", "translation_y", "rotation_deg", "scale"]

    def generate_points(
        self, n: int, rng: np.random.Generator, distribution: str, margin: float
    ) -> np.ndarray:
        lo, hi = margin, 1.0 - margin
        if distribution in ("uniform", "coord_uniform"):
            return rng.uniform(lo, hi, size=(n, 2))
        elif distribution == "clustered":
            n_clusters = rng.integers(2, 5)
            centers = rng.uniform(lo, hi, size=(n_clusters, 2))
            points = []
            for i in range(n):
                c = centers[i % n_clusters]
                spread = (hi - lo) * 0.08
                pt = rng.normal(c, spread, size=2)
                points.append(pt)
            pts = np.array(points)
            return np.clip(pts, lo, hi)
        elif distribution == "grid":
            side = int(np.ceil(np.sqrt(n)))
            xs = np.linspace(lo, hi, side)
            ys = np.linspace(lo, hi, side)
            grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)[:n]
            jitter = rng.uniform(-0.01, 0.01, size=grid.shape)
            return np.clip(grid + jitter, lo, hi)
        else:
            raise ValueError(f"Unknown distribution: {distribution!r}")

    def apply_transform(self, points: np.ndarray, transform_params: dict[str, float]) -> np.ndarray:
        pts = points.copy()
        centroid = pts.mean(axis=0)

        # Scale from centroid
        scale = transform_params.get("scale", 0.0)
        if scale != 0.0:
            factor = 1.0 + scale
            pts = (pts - centroid) * factor + centroid

        # Rotate around centroid
        rot_deg = transform_params.get("rotation_deg", 0.0)
        if rot_deg != 0.0:
            theta = np.radians(rot_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            centered = pts - centroid
            rotated = np.column_stack(
                [
                    centered[:, 0] * cos_t - centered[:, 1] * sin_t,
                    centered[:, 0] * sin_t + centered[:, 1] * cos_t,
                ]
            )
            pts = rotated + centroid

        # Translate
        tx = transform_params.get("translation_x", 0.0)
        ty = transform_params.get("translation_y", 0.0)
        pts[:, 0] += tx
        pts[:, 1] += ty

        return pts

    def to_image_coords(self, points: np.ndarray, image_size: int, margin: float) -> np.ndarray:
        margin_px = margin * image_size
        usable = image_size - 2 * margin_px
        pixel = points * usable + margin_px
        # Flip y-axis for image convention (y=0 at top)
        pixel[:, 1] = image_size - pixel[:, 1]
        return pixel

    def render_grid(self, ax, image_size: int, margin: float, color: str, alpha: float) -> None:
        margin_px = margin * image_size
        lo = margin_px
        hi = image_size - margin_px
        n_lines = 11
        for val in np.linspace(lo, hi, n_lines):
            ax.axhline(val, color=color, alpha=alpha, linewidth=0.5, zorder=0)
            ax.axvline(val, color=color, alpha=alpha, linewidth=0.5, zorder=0)

    def describe_transform(self, transform_params: dict[str, float]) -> str:
        parts = []
        tx = transform_params.get("translation_x", 0.0)
        ty = transform_params.get("translation_y", 0.0)
        rot = wrap_angle_deg(transform_params.get("rotation_deg", 0.0))
        scale = transform_params.get("scale", 0.0)
        if abs(tx) > 1e-6 or abs(ty) > 1e-6:
            parts.append(f"translation ({tx:.3f}, {ty:.3f})")
        if abs(rot) > 1e-6:
            parts.append(f"rotation {rot:.1f}°")
        if abs(scale) > 1e-6:
            parts.append(f"scale {1.0 + scale:.3f}x")
        return ", ".join(parts) if parts else "identity"

    def classify_direction(self, transform_params: dict[str, float]) -> str:
        tx = transform_params.get("translation_x", 0.0)
        ty = transform_params.get("translation_y", 0.0)
        rot = wrap_angle_deg(transform_params.get("rotation_deg", 0.0))
        scale = transform_params.get("scale", 0.0)

        # Pick the dominant transform component
        candidates: list[tuple[float, str]] = []
        if abs(tx) > 1e-6:
            candidates.append((abs(tx), "right" if tx > 0 else "left"))
        if abs(ty) > 1e-6:
            # Rendering flips y (pixel y = image_size - native y) and matplotlib
            # uses y-up axes, so positive ty (higher native y) renders visually DOWN
            candidates.append((abs(ty), "down" if ty > 0 else "up"))
        if abs(rot) > 1e-6:
            # Positive angle = CCW in math, but y-flip makes it CW in image
            candidates.append((abs(rot) / 180.0, "clockwise" if rot > 0 else "counterclockwise"))
        if abs(scale) > 1e-6:
            # Scale doesn't map cleanly to a direction; treat as outward/inward
            # but those aren't in Cartesian vocabulary. Use the largest other component.
            pass

        if not candidates:
            raise ValueError(
                "classify_direction: no direction-mappable transform component "
                f"in params={transform_params!r}. Cartesian 'scale' has no "
                "direction reading; categorical sampling must draw from "
                "direction_mappable_transforms() (sophon.questions)."
            )

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def transform_magnitude(self, transform_params: dict[str, float]) -> float:
        tx = transform_params.get("translation_x", 0.0)
        ty = transform_params.get("translation_y", 0.0)
        rot = wrap_angle_deg(transform_params.get("rotation_deg", 0.0))
        scale = transform_params.get("scale", 0.0)
        return np.sqrt(tx**2 + ty**2) + abs(rot) / 180.0 + abs(scale)

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
        # Describe the DISPLAY frame, not the native math frame: to_image_coords
        # flips y for image convention, and every direction ground-truth word
        # ("up"/"down", "clockwise"/"counterclockwise") is defined in the rendered
        # frame. Printing raw math-frame y reversed the trans_y and rotation GTs
        # in text_description mode (audited 2026-08-03: 8/8 inverted on both).
        # y' = 1 - y keeps the stated [0, 1] range and the x-based sort order.
        sorted_pts = points[points[:, 0].argsort()]
        lines = [
            "Points in a 2D Cartesian coordinate system with x \u2208 [0, 1] and y \u2208 [0, 1]:"
        ]
        for i, pt in enumerate(sorted_pts):
            lines.append(f"  Point {i + 1}: ({pt[0]:.3f}, {1.0 - pt[1]:.3f})")
        return "\n".join(lines)
