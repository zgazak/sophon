"""Tests for CLI batch evaluation flags (--specs-dir, --seeds)."""

from click.testing import CliRunner

from sophon.cli import _load_specs_from_dir, cli

MINIMAL_SPEC_YAML = """\
name: {name}
coordinate_system: cartesian_2d
points:
  num_points: [5, 10]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
questions:
  - type: binary
num_samples: 3
seed: 42
"""


class TestSpecsDir:
    def test_load_specs_from_dir(self, tmp_path):
        # Create two spec files
        for name in ["spec_a", "spec_b"]:
            (tmp_path / f"{name}.yaml").write_text(MINIMAL_SPEC_YAML.format(name=name))
        specs = _load_specs_from_dir(str(tmp_path))
        assert len(specs) == 2
        names = {s.name for s in specs}
        assert names == {"spec_a", "spec_b"}

    def test_load_specs_from_dir_ignores_non_specs(self, tmp_path):
        (tmp_path / "spec_a.yaml").write_text(MINIMAL_SPEC_YAML.format(name="spec_a"))
        (tmp_path / "README.md").write_text("Not a spec")
        (tmp_path / "notes.txt").write_text("Not a spec")
        specs = _load_specs_from_dir(str(tmp_path))
        assert len(specs) == 1


class TestSeedsFlag:
    def test_generate_with_different_seeds(self, tmp_path):
        """Different seeds should produce different ground truths."""
        import yaml

        from sophon.generator import BenchmarkGenerator
        from sophon.schema import BenchmarkSpec

        spec = BenchmarkSpec(**yaml.safe_load(MINIMAL_SPEC_YAML.format(name="seed_test")))
        gen = BenchmarkGenerator(spec)

        results_42 = [s.ground_truth for _, s in gen.generate(seed=42)]
        results_99 = [s.ground_truth for _, s in gen.generate(seed=99)]
        # Should differ (extremely high probability with different seeds)
        assert results_42 != results_99


class TestSystemPromptFlag:
    def test_spec_system_prompt(self):
        """Spec-level system_prompt field should be accepted."""
        import yaml

        from sophon.schema import BenchmarkSpec

        spec_yaml = MINIMAL_SPEC_YAML.format(name="sys_prompt_test")
        spec_yaml += "system_prompt: 'Think step by step.'\n"
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        assert spec.system_prompt == "Think step by step."


class TestGenerateCommand:
    def test_standard_output(self, tmp_path):
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(MINIMAL_SPEC_YAML.format(name="cli_test"))
        out_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "generate",
                "--spec",
                str(spec_path),
                "--output-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "metadata.jsonl").exists()
        assert (out_dir / "ground_truth.json").exists()
        assert (out_dir / "images").is_dir()

    def test_num_samples_override(self, tmp_path):
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(MINIMAL_SPEC_YAML.format(name="override_test"))
        out_dir = tmp_path / "output"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "generate",
                "--spec",
                str(spec_path),
                "--output-dir",
                str(out_dir),
                "--num-samples",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        # 2 samples x 1 question type = 2 lines
        lines = (out_dir / "metadata.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
