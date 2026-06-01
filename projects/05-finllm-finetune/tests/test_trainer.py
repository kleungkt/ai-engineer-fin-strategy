"""Tests for the trainer module — LoRA config and training logic."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch, call

import pytest

from src.trainer import LoRAConfig, TrainingConfig


# ---------------------------------------------------------------------------
# LoRAConfig
# ---------------------------------------------------------------------------

class TestLoRAConfig:
    def test_default_values(self):
        cfg = LoRAConfig()
        assert cfg.r == 16
        assert cfg.alpha == 32
        assert cfg.dropout == 0.05
        assert "q_proj" in cfg.target_modules
        assert cfg.bias == "none"

    def test_scaling_property(self):
        cfg = LoRAConfig(r=16, alpha=32)
        assert cfg.scaling == 2.0

    def test_scaling_custom(self):
        cfg = LoRAConfig(r=8, alpha=64)
        assert cfg.scaling == 8.0

    def test_custom_values(self):
        cfg = LoRAConfig(r=8, alpha=16, dropout=0.1, target_modules=["q_proj"], bias="all")
        assert cfg.r == 8
        assert cfg.alpha == 16
        assert cfg.dropout == 0.1
        assert cfg.bias == "all"

    def test_rank_validation(self):
        with pytest.raises(ValueError):
            LoRAConfig(r=0)

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            LoRAConfig(alpha=0)

    def test_dropout_bounds(self):
        with pytest.raises(ValueError):
            LoRAConfig(dropout=-0.1)
        with pytest.raises(ValueError):
            LoRAConfig(dropout=1.5)

    def test_model_dump(self):
        cfg = LoRAConfig(r=8, alpha=16)
        d = cfg.model_dump()
        assert d["r"] == 8
        assert d["alpha"] == 16
        LoRAConfig(**d)  # round-trip


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_default_values(self):
        cfg = TrainingConfig()
        assert cfg.base_model == "meta-llama/Llama-2-7b-hf"
        assert cfg.num_epochs == 3
        assert cfg.batch_size == 4
        assert cfg.learning_rate == 2e-4
        assert cfg.max_seq_length == 512
        assert cfg.use_4bit is True
        assert isinstance(cfg.lora_config, LoRAConfig)

    def test_custom_values(self, training_config_dict):
        cfg = TrainingConfig(**training_config_dict)
        assert cfg.base_model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert cfg.num_epochs == 1
        assert cfg.use_4bit is False

    def test_nested_lora_config(self):
        cfg = TrainingConfig(lora_config=LoRAConfig(r=32, alpha=64))
        assert cfg.lora_config.r == 32
        assert cfg.lora_config.alpha == 64

    def test_epoch_validation(self):
        with pytest.raises(ValueError):
            TrainingConfig(num_epochs=0)

    def test_batch_size_validation(self):
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)

    def test_learning_rate_validation(self):
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0)

    def test_max_seq_length_validation(self):
        with pytest.raises(ValueError):
            TrainingConfig(max_seq_length=16)

    def test_model_dump_round_trip(self, training_config_dict):
        cfg = TrainingConfig(**training_config_dict)
        d = cfg.model_dump()
        cfg2 = TrainingConfig(**d)
        assert cfg2 == cfg


# ---------------------------------------------------------------------------
# prepare_model (mocked via sys.modules patching)
# ---------------------------------------------------------------------------

class TestPrepareModel:
    def test_prepare_model_returns_tuple(self):
        """prepare_model should return (model, tokenizer, peft_config)."""
        # Create mock modules for the local imports inside prepare_model
        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        mock_transformers = ModuleType("transformers")
        mock_transformers.AutoModelForCausalLM = MagicMock()
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.BitsAndBytesConfig = MagicMock()

        mock_peft = ModuleType("peft")
        mock_peft.LoraConfig = MagicMock()
        mock_peft.get_peft_model = MagicMock()
        mock_peft.TaskType = MagicMock()
        mock_peft.TaskType.CAUSAL_LM = "CAUSAL_LM"

        # Set up tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        # Set up model mock
        mock_model = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

        # Set up peft model mock
        mock_peft_model = MagicMock()
        mock_peft.get_peft_model.return_value = mock_peft_model

        modules = {
            "torch": mock_torch,
            "transformers": mock_transformers,
            "peft": mock_peft,
        }

        with patch.dict(sys.modules, modules):
            # Re-import to pick up mocked modules
            import importlib
            import src.trainer as trainer_mod
            importlib.reload(trainer_mod)

            cfg = trainer_mod.TrainingConfig(use_4bit=True)
            result = trainer_mod.prepare_model(cfg)

            assert isinstance(result, tuple)
            assert len(result) == 3
            assert result[0] is mock_peft_model
            assert result[1] is mock_tokenizer

        # Reload back to original state
        importlib.reload(trainer_mod)

    def test_prepare_model_without_4bit(self):
        """prepare_model with use_4bit=False should not use BitsAndBytesConfig."""
        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        mock_transformers = ModuleType("transformers")
        mock_model_cls = MagicMock()
        mock_tokenizer_cls = MagicMock()
        mock_bnb_cls = MagicMock()
        mock_transformers.AutoModelForCausalLM = mock_model_cls
        mock_transformers.AutoTokenizer = mock_tokenizer_cls
        mock_transformers.BitsAndBytesConfig = mock_bnb_cls

        mock_peft = ModuleType("peft")
        mock_peft_lora = MagicMock()
        mock_peft_get = MagicMock()
        mock_peft_task = MagicMock()
        mock_peft_task.CAUSAL_LM = "CAUSAL_LM"
        mock_peft.LoraConfig = mock_peft_lora
        mock_peft.get_peft_model = mock_peft_get
        mock_peft.TaskType = mock_peft_task

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "</s>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_peft_get.return_value = mock_model

        modules = {
            "torch": mock_torch,
            "transformers": mock_transformers,
            "peft": mock_peft,
        }

        with patch.dict(sys.modules, modules):
            import importlib
            import src.trainer as trainer_mod
            importlib.reload(trainer_mod)

            cfg = trainer_mod.TrainingConfig(use_4bit=False)
            model, tokenizer, peft_config = trainer_mod.prepare_model(cfg)

            # Should have passed quantization_config=None
            call_kwargs = mock_model_cls.from_pretrained.call_args
            assert call_kwargs[1].get("quantization_config") is None

        importlib.reload(trainer_mod)


# ---------------------------------------------------------------------------
# train (mocked)
# ---------------------------------------------------------------------------

class TestTrain:
    def test_train_returns_output_path(self):
        """train() should return the adapter output path string."""
        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        mock_transformers = ModuleType("transformers")
        mock_training_args = MagicMock()
        mock_trainer_cls = MagicMock()
        mock_transformers.TrainingArguments = mock_training_args
        mock_transformers.Trainer = mock_trainer_cls
        mock_transformers.AutoModelForCausalLM = MagicMock()
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.BitsAndBytesConfig = MagicMock()

        mock_datasets = ModuleType("datasets")
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.column_names = ["instruction", "input", "output"]
        mock_hf_dataset.map.return_value = mock_hf_dataset
        mock_datasets.Dataset = MagicMock()
        mock_datasets.Dataset.from_list.return_value = mock_hf_dataset

        mock_peft = ModuleType("peft")
        mock_peft.LoraConfig = MagicMock()
        mock_peft.get_peft_model = MagicMock()
        mock_peft.TaskType = MagicMock()
        mock_peft.TaskType.CAUSAL_LM = "CAUSAL_LM"

        modules = {
            "torch": mock_torch,
            "transformers": mock_transformers,
            "datasets": mock_datasets,
            "peft": mock_peft,
        }

        with patch.dict(sys.modules, modules):
            import importlib
            import src.trainer as trainer_mod
            importlib.reload(trainer_mod)

            # Mock prepare_model to avoid actual model loading
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_peft_cfg = MagicMock()

            with patch.object(trainer_mod, "prepare_model", return_value=(mock_model, mock_tokenizer, mock_peft_cfg)):
                train_data = [
                    {"instruction": "test", "input": "in", "output": "out"},
                ]

                cfg = trainer_mod.TrainingConfig(output_dir="/tmp/test_train_output", num_epochs=1)
                result = trainer_mod.train(cfg, train_data)

                assert isinstance(result, str)
                assert "final_adapter" in result
                mock_trainer = mock_trainer_cls.return_value
                mock_trainer.train.assert_called_once()
                mock_model.save_pretrained.assert_called_once()

        importlib.reload(trainer_mod)

    def test_train_with_eval_data(self):
        """train() with eval_data should pass it to the Trainer."""
        mock_torch = MagicMock()
        mock_torch.bfloat16 = "bfloat16"

        mock_transformers = ModuleType("transformers")
        mock_transformers.TrainingArguments = MagicMock()
        mock_trainer_cls = MagicMock()
        mock_transformers.Trainer = mock_trainer_cls
        mock_transformers.AutoModelForCausalLM = MagicMock()
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.BitsAndBytesConfig = MagicMock()

        mock_datasets = ModuleType("datasets")
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.column_names = ["instruction", "input", "output"]
        mock_hf_dataset.map.return_value = mock_hf_dataset
        mock_datasets.Dataset = MagicMock()
        mock_datasets.Dataset.from_list.return_value = mock_hf_dataset

        mock_peft = ModuleType("peft")
        mock_peft.LoraConfig = MagicMock()
        mock_peft.get_peft_model = MagicMock()
        mock_peft.TaskType = MagicMock()
        mock_peft.TaskType.CAUSAL_LM = "CAUSAL_LM"

        modules = {
            "torch": mock_torch,
            "transformers": mock_transformers,
            "datasets": mock_datasets,
            "peft": mock_peft,
        }

        with patch.dict(sys.modules, modules):
            import importlib
            import src.trainer as trainer_mod
            importlib.reload(trainer_mod)

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_peft_cfg = MagicMock()

            with patch.object(trainer_mod, "prepare_model", return_value=(mock_model, mock_tokenizer, mock_peft_cfg)):
                train_data = [{"instruction": "t", "input": "i", "output": "o"}]
                eval_data = [{"instruction": "e", "input": "i", "output": "o"}]

                cfg = trainer_mod.TrainingConfig(output_dir="/tmp/test_eval")
                trainer_mod.train(cfg, train_data, eval_data=eval_data)

                # Dataset.from_list should be called twice (train + eval)
                assert mock_datasets.Dataset.from_list.call_count == 2

        importlib.reload(trainer_mod)
