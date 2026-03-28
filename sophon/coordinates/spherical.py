"""Spherical (fisheye) coordinate system."""

from __future__ import annotations

import numpy as np

from sophon.coordinates.base import CoordinateSystem
from sophon.coordinates.registry import register


@register("spherical")
class Spherical(CoordinateSystem):
    """Points on a hemisphere in (altitude, azimuth), projected via equidistant fisheye.

    Altitude: 0 = horizon, pi/2 = zenith.
    Azimuth: 0 = north (up in image), increasing clockwise.
    Fisheye projection: r = f * zenith_angle, where zenith_angle = pi/2 - altitude.
    """

    @property
    def name(self) -> str:
        return "spherical"

    @property
    def ndim(self) -> int:
        return 2

    @property
    def direction_vocabulary(self) -> list[str]:
        return [
            "rotate_azimuth_cw",
            "rotate_azimuth_ccw",
            "shift_zenith_in",
            "shift_zenith_out",
            "roll_cw",
            "roll_ccw",
        ]

    @property
    def supported_transforms(self) -> list[str]:
        return ["azimuth_rotation_deg", "altitude_offset_deg", "roll_deg"]

    def generate_points(
        self, n: int, rng: np.random.Generator, distribution: str, margin: float
    ) -> np.ndarray:
        # altitude in [margin_alt, pi/2 - margin_alt]
        # Weight by cos(altitude) for area-uniform on sphere
        alt_min = margin * (np.pi / 2)
        alt_max = np.pi / 2 - margin * (np.pi / 2)

        if distribution == "uniform":
            # Area-uniform: sample sin(alt) uniformly, then invert
            sin_lo = np.sin(alt_min)
            sin_hi = np.sin(alt_max)
            sin_alt = rng.uniform(sin_lo, sin_hi, size=n)
            altitude = np.arcsin(sin_alt)
            azimuth = rng.uniform(0, 2 * np.pi, size=n)
            return np.column_stack([altitude, azimuth])
        elif distribution == "coord_uniform":
            # Coordinate-uniform: raw uniform in (altitude, azimuth).
            # Uniform altitude → uniform projected radius → density ∝ 1/r
            # in image but with different falloff than polar coord_uniform.
            # Used for coord_id to reveal geometry.
            altitude = rng.uniform(alt_min, alt_max, size=n)
            azimuth = rng.uniform(0, 2 * np.pi, size=n)
            return np.column_stack([altitude, azimuth])
        elif distribution == "clustered":
            n_clusters = rng.integers(2, 5)
            c_alt = rng.uniform(alt_min + 0.1, alt_max - 0.1, size=n_clusters)
            c_az = rng.uniform(0, 2 * np.pi, size=n_clusters)
            points = []
            for i in range(n):
                ci = i % n_clusters
                alt = rng.normal(c_alt[ci], 0.1)
                az = rng.normal(c_az[ci], 0.3)
                alt = np.clip(alt, alt_min, alt_max)
                az = az % (2 * np.pi)
                points.append([alt, az])
            return np.array(points)
        elif distribution == "grid":
            n_alt = int(np.ceil(np.sqrt(n)))
            n_az = int(np.ceil(n / n_alt))
            alts = np.linspace(alt_min, alt_max, n_alt)
            azs = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
            grid = np.array([(a, z) for a in alts for z in azs])[:n]
            ja = rng.uniform(-0.03, 0.03, size=len(grid))
            jz = rng.uniform(-0.05, 0.05, size=len(grid))
            grid[:, 0] = np.clip(grid[:, 0] + ja, alt_min, alt_max)
            grid[:, 1] = (grid[:, 1] + jz) % (2 * np.pi)
            return grid
        else:
            raise ValueError(f"Unknown distribution: {distribution!r}")

    def apply_transform(
        self, points: np.ndarray, transform_params: dict[str, float]
    ) -> np.ndarray:
        pts = points.copy()
        altitude, azimuth = pts[:, 0], pts[:, 1]

        # Altitude offset
        alt_off = transform_params.get("altitude_offset_deg", 0.0)
        if alt_off != 0.0:
            altitude = np.clip(altitude + np.radians(alt_off), 0, np.pi / 2)

        # Azimuth rotation
        az_rot = transform_params.get("azimuth_rotation_deg", 0.0)
        if az_rot != 0.0:
            azimuth = (azimuth + np.radians(az_rot)) % (2 * np.pi)

        # Roll is handled as post-projection rotation (not in native space)
        # We store it in params but apply it in to_image_coords
        return np.column_stack([altitude, azimuth])

    def to_image_coords(
        self,
        points: np.ndarray,
        image_size: int,
        margin: float,
        roll_deg: float = 0.0,
    ) -> np.ndarray:
        altitude, azimuth = points[:, 0], points[:, 1]
        zenith_angle = np.pi / 2 - altitude

        # Equidistant fisheye: r_image = f * zenith_angle
        # f chosen so horizon (zenith_angle = pi/2) maps to edge of usable area
        margin_px = margin * image_size
        usable_radius = (image_size - 2 * margin_px) / 2.0
        f = usable_radius / (np.pi / 2)
        r_image = f * zenith_angle

        # Azimuth 0 = north = up in image, increasing clockwise
        # In image coords: x = center + r * sin(az), y = center - r * cos(az)
        center = image_size / 2.0
        x = center + r_image * np.sin(azimuth)
        y = center - r_image * np.cos(azimuth)

        # Apply roll as 2D rotation around image center
        if roll_deg != 0.0:
            theta = np.radians(roll_deg)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            dx, dy = x - center, y - center
            x = center + dx * cos_t - dy * sin_t
            y = center + dx * sin_t + dy * cos_t

        return np.column_stack([x, y])

    def render_grid(self, ax, image_size: int, margin: float, color: str, alpha: float) -> None:
        import matplotlib.patches as mpatches

        margin_px = margin * image_size
        usable_radius = (image_size - 2 * margin_px) / 2.0
        center = image_size / 2.0

        # Concentric altitude rings
        for frac in np.linspace(0.2, 1.0, 5):
            circle = mpatches.Circle(
                (center, center), frac * usable_radius,
                fill=False, edgecolor=color, alpha=alpha, linewidth=0.5, zorder=0,
            )
            ax.add_patch(circle)
        # Radial azimuth lines
        for angle_deg in range(0, 360, 30):
            angle = np.radians(angle_deg)
            x_end = center + usable_radius * np.sin(angle)
            y_end = center - usable_radius * np.cos(angle)
            ax.plot([center, x_end], [center, y_end],
                    color=color, alpha=alpha, linewidth=0.5, zorder=0)

    def describe_transform(self, transform_params: dict[str, float]) -> str:
        parts = []
        az = transform_params.get("azimuth_rotation_deg", 0.0)
        alt = transform_params.get("altitude_offset_deg", 0.0)
        roll = transform_params.get("roll_deg", 0.0)
        if abs(az) > 1e-6:
            parts.append(f"azimuth rotation {az:.1f}°")
        if abs(alt) > 1e-6:
            parts.append(f"altitude offset {alt:.1f}°")
        if abs(roll) > 1e-6:
            parts.append(f"roll {roll:.1f}°")
        return ", ".join(parts) if parts else "identity"

    def classify_direction(self, transform_params: dict[str, float]) -> str:
        az = transform_params.get("azimuth_rotation_deg", 0.0)
        alt = transform_params.get("altitude_offset_deg", 0.0)
        roll = transform_params.get("roll_deg", 0.0)

        candidates: list[tuple[float, str]] = []
        if abs(az) > 1e-6:
            # Positive azimuth = clockwise (spec convention)
            candidates.append(
                (abs(az) / 180.0, "rotate_azimuth_cw" if az > 0 else "rotate_azimuth_ccw")
            )
        if abs(alt) > 1e-6:
            # Positive altitude offset = toward zenith = toward center of image
            candidates.append(
                (abs(alt) / 90.0, "shift_zenith_in" if alt > 0 else "shift_zenith_out")
            )
        if abs(roll) > 1e-6:
            candidates.append(
                (abs(roll) / 180.0, "roll_cw" if roll > 0 else "roll_ccw")
            )

        if not candidates:
            return "rotate_azimuth_cw"  # fallback

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def transform_magnitude(self, transform_params: dict[str, float]) -> float:
        az = transform_params.get("azimuth_rotation_deg", 0.0)
        alt = transform_params.get("altitude_offset_deg", 0.0)
        roll = transform_params.get("roll_deg", 0.0)
        return abs(az) / 180.0 + abs(alt) / 90.0 + abs(roll) / 180.0

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
