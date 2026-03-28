# Sophon

Generative benchmark for probing spatial reasoning in Vision-Language Models across coordinate geometries.

Sophon ships no images — it ships a generator. A YAML spec + integer seed deterministically produces (image, question, ground_truth) triples for evaluating how well VLMs understand coordinate geometry.

## Quick start

```bash
pip install sophon-bench
sophon preview --spec sophon/specs/cartesian_rotation.yaml --num-samples 5
```

## Coordinate systems (v0.1)

- **1D Linear** — points on a number line, translation only
- **2D Cartesian** — points on a grid, translation/rotation/scale
- **Polar** — points in (r, θ) space, radial shift/angular rotation/radial scale

## Question types

- **Binary** — "Are the point sets aligned to within one dot width?" (yes/no)
- **Categorical** — "Which direction should the red points move?" (e.g., left, clockwise)
- **Comparative** — side-by-side A|B, "Which shows better alignment?"
- **Quantitative** — "Estimate the rotation angle in degrees" (magnitude only)
- **Coordinate identification** — "What coordinate system are these points in?" (3-way)

## Future directions (v0.2+)

- **Spherical (fisheye)** — hemisphere projected via equidistant fisheye. Motivated by all-sky camera alignment (see `allclear` project). Deferred from v0.1 because polar and spherical are visually near-identical with abstract dot patterns; spherical needs more distinctive visual treatment to add diagnostic value.
- **Projective (3D perspective)** — depth encoding with uniform-size dots is visually ambiguous. Needs point-size or occlusion conventions that don't conflate the encoding with the spatial reasoning being tested.

## License

MIT
