"""Tests for the data pipeline module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data_pipeline import (
    FinDataSample,
    format_for_training,
    generate_sample_dataset,
    load_raw_data,
    save_dataset,
    split_dataset,
)


# ---------------------------------------------------------------------------
# FinDataSample model
# ---------------------------------------------------------------------------

class TestFinDataSample:
    """Tests for the Pydantic data model."""

    def test_valid_sample_creation(self):
        s = FinDataSample(
            instruction="test instruction",
            input="test input",
            output="test output",
            category="sentiment",
        )
        assert s.category == "sentiment"
        assert s.instruction == "test instruction"

    def test_all_valid_categories(self):
        for cat in ("sentiment", "qa", "summary", "analysis"):
            s = FinDataSample(instruction="i", input="x", output="y", category=cat)
            assert s.category == cat

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Category must be one of"):
            FinDataSample(instruction="i", input="x", output="y", category="invalid")

    def test_model_dump_round_trip(self):
        s = FinDataSample(instruction="i", input="x", output="y", category="qa")
        d = s.model_dump()
        s2 = FinDataSample(**d)
        assert s2 == s


# ---------------------------------------------------------------------------
# load_raw_data
# ---------------------------------------------------------------------------

class TestLoadRawData:
    def test_load_jsonl(self, sample_data_path):
        samples = load_raw_data(sample_data_path)
        assert len(samples) == 4
        assert all(isinstance(s, FinDataSample) for s in samples)

    def test_load_json(self, sample_json_path):
        samples = load_raw_data(sample_json_path)
        assert len(samples) == 4

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data("/nonexistent/path.json")

    def test_unsupported_format_raises(self, tmp_path):
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("a,b,c")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_raw_data(str(bad_file))


# ---------------------------------------------------------------------------
# generate_sample_dataset
# ---------------------------------------------------------------------------

class TestGenerateSampleDataset:
    def test_default_count(self):
        samples = generate_sample_dataset()
        assert len(samples) == 100

    def test_custom_count(self):
        samples = generate_sample_dataset(n=20)
        assert len(samples) == 20

    def test_small_count(self):
        samples = generate_sample_dataset(n=4)
        assert len(samples) == 4

    def test_all_categories_present(self):
        samples = generate_sample_dataset(n=100)
        categories = {s.category for s in samples}
        assert categories == {"sentiment", "qa", "summary", "analysis"}

    def test_samples_are_valid(self):
        samples = generate_sample_dataset(n=12)
        for s in samples:
            assert isinstance(s, FinDataSample)
            assert s.instruction
            assert s.output


# ---------------------------------------------------------------------------
# format_for_training
# ---------------------------------------------------------------------------

class TestFormatForTraining:
    def test_alpaca_format(self, sample_data):
        formatted = format_for_training(sample_data, format="alpaca")
        assert len(formatted) == 4
        assert all(k in formatted[0] for k in ("instruction", "input", "output"))
        assert "messages" not in formatted[0]

    def test_chatml_format(self, sample_data):
        formatted = format_for_training(sample_data, format="chatml")
        assert len(formatted) == 4
        assert "messages" in formatted[0]
        msgs = formatted[0]["messages"]
        assert len(msgs) == 3
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user", "assistant"]

    def test_default_format_is_alpaca(self, sample_data):
        formatted = format_for_training(sample_data)
        assert "instruction" in formatted[0]

    def test_invalid_format_raises(self, sample_data):
        with pytest.raises(ValueError, match="Unsupported format"):
            format_for_training(sample_data, format="invalid")


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------

class TestSplitDataset:
    def test_default_split(self, sample_data):
        train, test = split_dataset(sample_data, test_ratio=0.25)
        assert len(train) + len(test) == 4
        # With 4 items and 0.25 ratio, 1 test and 3 train
        assert len(test) >= 1

    def test_all_train(self, sample_data):
        train, test = split_dataset(sample_data, test_ratio=0.0)
        assert len(train) == 4
        assert len(test) == 0

    def test_all_test(self, sample_data):
        train, test = split_dataset(sample_data, test_ratio=1.0)
        # split_idx = max(1, int(4 * 0)) = 1 => 1 train, 3 test
        assert len(train) >= 1

    def test_invalid_ratio_raises(self, sample_data):
        with pytest.raises(ValueError, match="test_ratio must be between"):
            split_dataset(sample_data, test_ratio=1.5)

    def test_negative_ratio_raises(self, sample_data):
        with pytest.raises(ValueError, match="test_ratio must be between"):
            split_dataset(sample_data, test_ratio=-0.1)

    def test_works_with_dicts(self):
        data = [{"a": i} for i in range(10)]
        train, test = split_dataset(data, test_ratio=0.2)
        assert len(train) + len(test) == 10


# ---------------------------------------------------------------------------
# save_dataset
# ---------------------------------------------------------------------------

class TestSaveDataset:
    def test_save_fin_data_samples(self, tmp_path, sample_data):
        out = tmp_path / "out.jsonl"
        save_dataset(sample_data, str(out))
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 4
        first = json.loads(lines[0])
        assert "instruction" in first

    def test_save_dicts(self, tmp_path, sample_formatted_alpaca):
        out = tmp_path / "dicts.jsonl"
        save_dataset(sample_formatted_alpaca, str(out))
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_creates_parent_dirs(self, tmp_path, sample_data):
        out = tmp_path / "sub" / "dir" / "out.jsonl"
        save_dataset(sample_data, str(out))
        assert out.exists()
