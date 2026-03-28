"""Tests for the renderer module."""

import numpy as np
from PIL import Image, ImageChops

from sophon.coordinates import get_coordinate_system
from sophon.renderer import render_comparative, render_sample
from sophon.schema import RenderingConfig


def make_rng(seed=42):
    return np.random.Generator(np.random.PCG64(seed))


def _generate_test_data(coord_name, seed=42):
    """Generate reference + transformed points for testing."""
    cs = get_coordinate_system(coord_name)
    rng = make_rng(seed)
    ref = cs.generate_points(20, rng, "uniform", 0.1)
    ranges = {k: [-0.1, 0.1] for k in cs.supported_transforms}
    params = cs.sample_transform(rng, ranges)
    trans = cs.apply_transform(ref, params)
    return cs, ref, trans, params


class TestRenderSample:
    def test_output_is_pil_image(self):
        cs, ref, trans, params = _generate_test_data("cartesian_2d")
        img = render_sample(cs, ref, trans, params, RenderingConfig(), 0.1)
        assert isinstance(img, Image.Image)

    def test_output_size(self):
        cs, ref, trans, params = _generate_test_data("cartesian_2d")
        config = RenderingConfig(image_size=256)
        img = render_sample(cs, ref, trans, params, config, 0.1)
        assert img.size[0] == img.size[1]  # Square

    def test_determinism(self):
        cs, ref, trans, params = _generate_test_data("cartesian_2d")
        config = RenderingConfig()
        img1 = render_sample(cs, ref, trans, params, config, 0.1)
        img2 = render_sample(cs, ref, trans, params, config, 0.1)
        diff = ImageChops.difference(img1, img2)
        assert diff.getbbox() is None, "Identical inputs should produce identical images"

    def test_all_coordinate_systems(self):
        for name in ["linear_1d", "cartesian_2d", "polar", "spherical"]:
            cs, ref, trans, params = _generate_test_data(name)
            img = render_sample(cs, ref, trans, params, RenderingConfig(), 0.1)
            assert isinstance(img, Image.Image)

    def test_no_grid(self):
        cs, ref, trans, params = _generate_test_data("cartesian_2d")
        config = RenderingConfig(show_grid=False)
        img = render_sample(cs, ref, trans, params, config, 0.1)
        assert isinstance(img, Image.Image)


class TestRenderComparative:
    def test_double_width(self):
        cs, ref, trans, params = _generate_test_data("cartesian_2d")
        rng = make_rng(99)
        ranges = {k: [-0.2, 0.2] for k in cs.supported_transforms}
        params_b = cs.sample_transform(rng, ranges)
        trans_b = cs.apply_transform(ref, params_b)
        config = RenderingConfig()
        img = render_comparative(
            cs, ref, trans, trans_b, params, params_b, config, 0.1
        )
        assert isinstance(img, Image.Image)
        # Comparative should be wider than tall
        assert img.size[0] > img.size[1]

    def test_all_coordinate_systems(self):
        for name in ["linear_1d", "cartesian_2d", "polar", "spherical"]:
            cs, ref, trans, params = _generate_test_data(name)
            rng = make_rng(99)
            ranges = {k: [-0.1, 0.1] for k in cs.supported_transforms}
            params_b = cs.sample_transform(rng, ranges)
            trans_b = cs.apply_transform(ref, params_b)
            img = render_comparative(
                cs, ref, trans, trans_b, params, params_b, RenderingConfig(), 0.1
            )
            assert isinstance(img, Image.Image)
