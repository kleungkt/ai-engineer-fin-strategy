"""Shared fixtures for financial LLM fine-tune tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the project src is importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@pytest.fixture()
def sample_data():
    """Return a small list of FinDataSample objects covering all categories."""
    from src.data_pipeline import FinDataSample

    return [
        FinDataSample(
            instruction="Analyze the sentiment of the following financial news headline.",
            input="Apple reported record earnings",
            output="The sentiment is bullish.",
            category="sentiment",
        ),
        FinDataSample(
            instruction="Answer the following financial question.",
            input="What is a P/E ratio?",
            output="The P/E ratio compares stock price to earnings per share.",
            category="qa",
        ),
        FinDataSample(
            instruction="Summarize the financial report.",
            input="Revenue $10B, Net Income $2B, up 15% YoY.",
            output="Strong results with revenue of $10B and net income of $2B, up 15%.",
            category="summary",
        ),
        FinDataSample(
            instruction="Provide a financial analysis.",
            input="Analyze a covered call strategy.",
            output="A covered call generates income but caps upside.",
            category="analysis",
        ),
    ]


@pytest.fixture()
def sample_data_path(tmp_path, sample_data):
    """Write sample data to a JSONL file and return its path."""
    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for s in sample_data:
            f.write(json.dumps(s.model_dump(), ensure_ascii=False) + "\n")
    return str(path)


@pytest.fixture()
def sample_json_path(tmp_path, sample_data):
    """Write sample data to a JSON file and return its path."""
    path = tmp_path / "train.json"
    with open(path, "w") as f:
        json.dump([s.model_dump() for s in sample_data], f)
    return str(path)


@pytest.fixture()
def sample_formatted_alpaca():
    """Return pre-formatted alpaca-style data."""
    return [
        {"instruction": "Analyze sentiment", "input": "Stock up 10%", "output": "Bullish"},
        {"instruction": "Explain P/E", "input": "What is P/E?", "output": "Price to earnings ratio"},
    ]


@pytest.fixture()
def sample_formatted_chatml():
    """Return pre-formatted chatml-style data."""
    return [
        {
            "messages": [
                {"role": "system", "content": "Analyze sentiment"},
                {"role": "user", "content": "Stock up 10%"},
                {"role": "assistant", "content": "Bullish"},
            ]
        }
    ]


@pytest.fixture()
def training_config_dict():
    """Return a valid TrainingConfig dictionary."""
    return {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "num_epochs": 1,
        "batch_size": 2,
        "learning_rate": 3e-4,
        "max_seq_length": 256,
        "output_dir": "/tmp/test_output",
        "use_4bit": False,
    }
