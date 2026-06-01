"""LoRA fine-tuning configuration and training for financial LLMs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LoRAConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) hyperparameters."""

    r: int = Field(default=16, description="LoRA rank", ge=1)
    alpha: int = Field(default=32, description="LoRA alpha scaling factor", ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Modules to apply LoRA to",
    )
    bias: str = Field(default="none", description="Bias type: none, all, or lora_only")

    @property
    def scaling(self) -> float:
        """Compute the LoRA scaling factor alpha / r."""
        return self.alpha / self.r


class TrainingConfig(BaseModel):
    """Full training configuration."""

    base_model: str = Field(
        default="meta-llama/Llama-2-7b-hf",
        description="HuggingFace model identifier or local path",
    )
    lora_config: LoRAConfig = Field(default_factory=LoRAConfig)
    num_epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    max_seq_length: int = Field(default=512, ge=32)
    output_dir: str = Field(default="./output")
    use_4bit: bool = Field(default=True, description="Enable 4-bit quantization (QLoRA)")


def prepare_model(config: TrainingConfig):
    """Load a base model and attach LoRA adapters.

    Args:
        config: Training configuration containing model and LoRA settings.

    Returns:
        Tuple of (model, tokenizer, peft_config).

    Note:
        Requires ``transformers``, ``peft``, ``torch``, and optionally
        ``bitsandbytes`` for 4-bit quantization.  These are imported lazily
        so the module can be tested without GPU dependencies.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
    except ImportError as exc:
        raise ImportError(
            "Install transformers, peft, and torch to use prepare_model: "
            f"{exc}"
        ) from exc

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA / PEFT config
    peft_config = PeftLoraConfig(
        r=config.lora_config.r,
        lora_alpha=config.lora_config.alpha,
        lora_dropout=config.lora_config.dropout,
        target_modules=config.lora_config.target_modules,
        bias=config.lora_config.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer, peft_config


def train(
    config: TrainingConfig,
    train_data: list[dict],
    eval_data: list[dict] | None = None,
) -> str:
    """Run LoRA fine-tuning.

    Args:
        config: Training configuration.
        train_data: Training examples (list of dicts with ``input_ids`` etc. or
            raw text dicts that will be tokenized).
        eval_data: Optional evaluation dataset.

    Returns:
        Path to the directory containing the saved adapters.

    Note:
        This function requires GPU hardware.  For testing, mock
        ``prepare_model`` and the ``Trainer`` class.
    """
    try:
        from transformers import TrainingArguments, Trainer
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "Install transformers and datasets to use train: " f"{exc}"
        ) from exc

    model, tokenizer, peft_config = prepare_model(config)

    # Build HF Datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None

    # Tokenization helper
    def tokenize_fn(examples):
        texts = [
            f"{inst}\n{inp}\n{out}"
            for inst, inp, out in zip(
                examples.get("instruction", [""] * len(examples["input"])),
                examples["input"],
                examples["output"],
            )
        ]
        return tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
        )

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    if eval_dataset:
        eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting training for %d epochs …", config.num_epochs)
    trainer.train()

    # Save adapters
    output_path = Path(config.output_dir) / "final_adapter"
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    logger.info("Adapters saved to %s", output_path)
    return str(output_path)
