"""Tests for conversation output format."""

import json

import yaml
from click.testing import CliRunner

from sophon.cli import cli
from sophon.schema import BenchmarkSpec


class TestConversationFormat:
    def test_conversation_spec_field(self):
        spec_yaml = """\
name: conv_test
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
output_format: conversation
"""
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        assert spec.output_format == "conversation"

    def test_standard_is_default(self):
        spec_yaml = """\
name: std_test
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
        spec = BenchmarkSpec(**yaml.safe_load(spec_yaml))
        assert spec.output_format == "standard"

    def test_generate_conversation_output(self, tmp_path):
        """Test that conversation format generates conversations.jsonl."""
        spec_yaml = """\
name: conv_gen_test
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
output_format: conversation
"""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(spec_yaml)
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

        conv_path = out_dir / "conversations.jsonl"
        assert conv_path.exists()

        lines = conv_path.read_text().strip().split("\n")
        assert len(lines) == 3

        for line in lines:
            entry = json.loads(line)
            assert "id" in entry
            assert "conversations" in entry
            assert len(entry["conversations"]) == 2
            assert entry["conversations"][0]["role"] == "user"
            assert entry["conversations"][1]["role"] == "assistant"

    def test_conversation_binary_answers(self, tmp_path):
        """Binary answers should be terse 'Yes' or 'No'."""
        spec_yaml = """\
name: conv_binary_test
coordinate_system: cartesian_2d
points:
  num_points: [5, 10]
  margin: 0.1
transforms:
  translation_x: [-0.15, 0.15]
questions:
  - type: binary
num_samples: 10
seed: 42
output_format: conversation
"""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(spec_yaml)
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
        assert result.exit_code == 0

        conv_path = out_dir / "conversations.jsonl"
        for line in conv_path.read_text().strip().split("\n"):
            entry = json.loads(line)
            assistant = entry["conversations"][1]["content"]
            assert assistant in ("Yes", "No")

    def test_conversation_also_writes_standard(self, tmp_path):
        """Conversation format should also write metadata.jsonl and ground_truth.json."""
        spec_yaml = """\
name: conv_std_test
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
output_format: conversation
"""
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(spec_yaml)
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
        assert result.exit_code == 0
        assert (out_dir / "metadata.jsonl").exists()
        assert (out_dir / "ground_truth.json").exists()
