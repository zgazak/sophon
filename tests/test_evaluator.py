"""Tests for evaluation and scoring infrastructure."""

import json

from sophon.evaluator import (
    _api_key_error_message,
    _is_correct,
    _is_local,
    _parse_response,
    _quantitative_error,
    aggregate_results,
    resolve_api_key,
    score_predictions,
)


class TestApiKeyResolution:
    def test_local_no_key(self):
        assert _is_local("http://localhost:8000/v1")
        assert _is_local("http://127.0.0.1:8000/v1")
        assert not _is_local("https://api.openai.com/v1")
        assert not _is_local(None)

    def test_resolve_local_returns_none(self):
        key = resolve_api_key("http://localhost:8000/v1")
        assert key is None

    def test_error_message_openrouter(self):
        msg = _api_key_error_message("https://openrouter.ai/api/v1")
        assert "OPENROUTER_API_KEY" in msg

    def test_error_message_generic(self):
        msg = _api_key_error_message("https://api.example.com/v1")
        assert "OPENAI_API_KEY" in msg
        assert "SOPHON_API_KEY" in msg


class TestParseResponse:
    def test_binary(self):
        result = _parse_response("binary", "Yes, they appear aligned")
        assert result.parsed_answer == "yes"
        assert result.parse_success is True

    def test_categorical(self):
        vocab = ["left", "right", "up", "down"]
        result = _parse_response("categorical", "Move them to the left", vocab)
        assert result.parsed_answer == "left"

    def test_comparative(self):
        result = _parse_response("comparative", "Image A shows better alignment")
        assert result.parsed_answer == "A"

    def test_quantitative(self):
        result = _parse_response("quantitative", "About 15.5 degrees")
        assert result.parsed_answer == 15.5

class TestScoring:
    def test_is_correct(self):
        assert _is_correct("binary", "yes", "yes") is True
        assert _is_correct("binary", "no", "yes") is False
        assert _is_correct("binary", None, "yes") is None

    def test_quantitative_error(self):
        assert _quantitative_error(15.0, 20.0) == 5.0
        assert _quantitative_error(None, 20.0) is None

    def test_aggregate_results(self):
        predictions = [
            {
                "sample_id": 0,
                "coordinate_system": "cartesian_2d",
                "question_type": "binary",
                "ground_truth": "yes",
                "raw_response": "Yes",
                "parsed_answer": "yes",
                "parse_success": True,
                "correct": True,
                "error": None,
                "transform_params": {"translation_x": 0.01},
                "transform_magnitude": 0.01,
                "transform_direction": None,
                "num_points": 10,
                "show_grid": True,
            },
            {
                "sample_id": 1,
                "coordinate_system": "cartesian_2d",
                "question_type": "binary",
                "ground_truth": "no",
                "raw_response": "No",
                "parsed_answer": "no",
                "parse_success": True,
                "correct": True,
                "error": None,
                "transform_params": {"translation_x": 0.15},
                "transform_magnitude": 0.15,
                "transform_direction": None,
                "num_points": 10,
                "show_grid": True,
            },
        ]
        results = aggregate_results(predictions, "test_spec", "gpt-4o", 42, 2)
        assert results["metadata"]["spec_name"] == "test_spec"
        assert results["metadata"]["model_name"] == "gpt-4o"
        assert results["metadata"]["parse_rate"] == 1.0
        assert results["results"]["cartesian_2d"]["binary"]["accuracy"] == 1.0
        assert results["results"]["cartesian_2d"]["binary"]["chance"] == 0.5
        assert results["chance_baselines"]["binary"] == 0.5


class TestScorePredictions:
    def test_score_from_files(self, tmp_path):
        # Write predictions JSONL
        preds = [
            {
                "sample_id": 0,
                "coordinate_system": "cartesian_2d",
                "question_type": "binary",
                "ground_truth": "yes",
                "raw_response": "Yes",
                "parsed_answer": "yes",
                "parse_success": True,
                "correct": True,
                "error": None,
            },
        ]
        pred_path = tmp_path / "predictions.jsonl"
        with open(pred_path, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")

        # Write ground truth JSON
        gt = {
            "0": {
                "question_type": "binary",
                "ground_truth": "yes",
                "coordinate_system": "cartesian_2d",
            }
        }
        gt_path = tmp_path / "ground_truth.json"
        with open(gt_path, "w") as f:
            json.dump(gt, f)

        results = score_predictions(pred_path, gt_path)
        assert "results" in results
        assert "metadata" in results
