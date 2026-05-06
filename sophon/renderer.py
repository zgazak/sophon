"""Render point sets to PIL Images using matplotlib."""

from __future__ import annotations

import io

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sophon.coordinates.base import CoordinateSystem
from sophon.schema import RenderingConfig

matplotlib.use("Agg")


def render_sample(
    coord_system: CoordinateSystem,
    reference_points: np.ndarray,
    transformed_points: np.ndarray | None,
    transform_params: dict[str, float],
    rendering_config: RenderingConfig,
    margin: float,
) -> Image.Image:
    """Render a single image with blue reference and red transformed points.

    If transformed_points is None, renders only the reference set in a
    neutral color.
    """
    size = rendering_config.image_size
    dpi = 100
    fig, ax = plt.subplots(figsize=(size / dpi, size / dpi), dpi=dpi)

    _setup_axes(ax, size, rendering_config)

    if rendering_config.show_grid:
        coord_system.render_grid(
            ax, size, margin, rendering_config.grid_color, rendering_config.grid_alpha
        )

    dot_size = rendering_config.point_radius**2

    if transformed_points is None:
        # Single-set rendering (coord identification): neutral color, no offset
        ref_img = coord_system.to_image_coords(reference_points, size, margin)
        _plot_points(ax, ref_img, "#7C3AED", dot_size)  # Purple, neutral
        return _fig_to_image(fig, size)

    # Project to image coordinates
    ref_img = coord_system.to_image_coords(reference_points, size, margin)
    # Spherical needs roll_deg passed through
    roll_deg = transform_params.get("roll_deg", 0.0)
    if hasattr(coord_system.to_image_coords, "__code__") and "roll_deg" in (
        coord_system.to_image_coords.__code__.co_varnames
    ):
        trans_img = coord_system.to_image_coords(
            transformed_points, size, margin, roll_deg=roll_deg
        )
    else:
        trans_img = coord_system.to_image_coords(transformed_points, size, margin)

    # For 1D linear: offset blue above line, red below
    y_offset = 0.0
    if coord_system.ndim == 1:
        y_offset = size * 0.04

    _plot_points(ax, ref_img, rendering_config.color_a, dot_size, y_offset=-y_offset)
    _plot_points(ax, trans_img, rendering_config.color_b, dot_size, y_offset=y_offset)

    return _fig_to_image(fig, size)


def render_comparative(
    coord_system: CoordinateSystem,
    reference_points: np.ndarray,
    transformed_a: np.ndarray,
    transformed_b: np.ndarray,
    transform_params_a: dict[str, float],
    transform_params_b: dict[str, float],
    rendering_config: RenderingConfig,
    margin: float,
) -> Image.Image:
    """Render side-by-side A|B comparison image (double width)."""
    size = rendering_config.image_size
    dpi = 100
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(size * 2 / dpi, size / dpi), dpi=dpi)

    for ax, label, transformed, t_params in [
        (ax_a, "A", transformed_a, transform_params_a),
        (ax_b, "B", transformed_b, transform_params_b),
    ]:
        _setup_axes(ax, size, rendering_config)

        if rendering_config.show_grid:
            coord_system.render_grid(
                ax, size, margin, rendering_config.grid_color, rendering_config.grid_alpha
            )

        ref_img = coord_system.to_image_coords(reference_points, size, margin)
        roll_deg = t_params.get("roll_deg", 0.0)
        if hasattr(coord_system.to_image_coords, "__code__") and "roll_deg" in (
            coord_system.to_image_coords.__code__.co_varnames
        ):
            trans_img = coord_system.to_image_coords(transformed, size, margin, roll_deg=roll_deg)
        else:
            trans_img = coord_system.to_image_coords(transformed, size, margin)

        dot_size = rendering_config.point_radius**2
        y_offset = size * 0.04 if coord_system.ndim == 1 else 0.0
        _plot_points(ax, ref_img, rendering_config.color_a, dot_size, y_offset=-y_offset)
        _plot_points(ax, trans_img, rendering_config.color_b, dot_size, y_offset=y_offset)

        # Panel label
        ax.text(
            size * 0.05,
            size * 0.08,
            label,
            fontsize=16,
            fontweight="bold",
            color="#333333",
            zorder=10,
        )

    plt.subplots_adjust(wspace=0.02, left=0, right=1, top=1, bottom=0)
    return _fig_to_image(fig, size, width_multiplier=2)


def _setup_axes(ax, size: int, config: RenderingConfig) -> None:
    """Configure axes for pixel-coordinate rendering."""
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor(config.background_color)


def _plot_points(
    ax, img_coords: np.ndarray, color: str, dot_size: float, y_offset: float = 0.0
) -> None:
    """Scatter plot points in pixel coordinates."""
    x = img_coords[:, 0]
    y = img_coords[:, 1] + y_offset
    ax.scatter(x, y, s=dot_size, c=color, alpha=0.8, edgecolors="none", zorder=5)


def render_preview(
    plot_image: Image.Image,
    question_text: str,
    ground_truth: str | float,
    sample_id: int,
    question_type: str,
) -> Image.Image:
    """Composite a plot image with question/answer text below it for visual inspection."""
    from PIL import ImageDraw, ImageFont

    plot_w, plot_h = plot_image.size
    padding = 12
    line_height = 18
    # Wrap text to fit image width (rough character estimate)
    chars_per_line = max(plot_w // 8, 40)

    lines = [f"[{sample_id}] {question_type}"]
    # Word-wrap the question
    q_prefix = "Q: "
    _wrap_lines(lines, q_prefix + question_text, chars_per_line)
    gt_str = f"{ground_truth:.4f}" if isinstance(ground_truth, float) else str(ground_truth)
    lines.append(f"A: {gt_str}")

    text_height = len(lines) * line_height + padding * 2
    total_h = plot_h + text_height

    canvas = Image.new("RGB", (plot_w, total_h), "#F8F8F8")
    canvas.paste(plot_image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    except (OSError, IOError):
        font = ImageFont.load_default()

    y = plot_h + padding
    for i, line in enumerate(lines):
        color = "#555555" if i == 0 else "#222222"
        draw.text((padding, y), line, fill=color, font=font)
        y += line_height

    return canvas


def _wrap_lines(lines: list[str], text: str, max_chars: int) -> None:
    """Word-wrap text into lines list."""
    words = text.split()
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)


def _fig_to_image(fig, size: int, width_multiplier: int = 1) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    fig.patch.set_facecolor("#F8F8F8")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img.load()  # Force load before buffer is GC'd
    buf.close()
    return img
