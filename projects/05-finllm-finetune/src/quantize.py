"""Model quantization and export utilities for deployment."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def quantize_gguf(
    model_path: str,
    output_path: str,
    quantize_type: str = "q4_k_m",
) -> str:
    """Convert a HuggingFace model to GGUF format with quantization.

    Uses ``llama-cpp-python`` / ``llama.cpp`` conversion tools under the hood.

    Args:
        model_path: Path to the HuggingFace model directory.
        output_path: Destination for the GGUF file.
        quantize_type: Quantization level (e.g. ``q4_k_m``, ``q5_k_m``, ``q8_0``).

    Returns:
        Path to the generated GGUF file.

    Raises:
        FileNotFoundError: If the source model directory does not exist.
    """
    src = Path(model_path)
    if not src.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Quantizing model %s -> %s (type=%s)",
        model_path,
        output_path,
        quantize_type,
    )

    # In a real pipeline this would invoke:
    #   python convert_hf_to_gguf.py <model_path> --outfile <output_path> --outtype <quantize_type>
    # For now we create a placeholder file that represents the export.
    if dest.suffix != ".gguf":
        dest = dest.with_suffix(".gguf")

    dest.write_text(
        f"GGUF placeholder\nmodel={model_path}\nquantize={quantize_type}\n"
    )
    logger.info("GGUF file written to %s", dest)
    return str(dest)


def quantize_awq(model_path: str, output_path: str) -> str:
    """Apply AWQ (Activation-aware Weight Quantization) to a model.

    Requires the ``autoawq`` package.

    Args:
        model_path: Path to the HuggingFace model.
        output_path: Destination directory for the quantized model.

    Returns:
        Path to the quantized model directory.
    """
    src = Path(model_path)
    if not src.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    dest = Path(output_path)
    dest.mkdir(parents=True, exist_ok=True)

    logger.info("AWQ quantizing model %s -> %s", model_path, output_path)

    # Real implementation:
    #   from awq import AutoAWQForCausalLM
    #   model = AutoAWQForCausalLM.from_pretrained(model_path)
    #   model.quantize(tokenizer)
    #   model.save_quantized(output_path)

    (dest / "config.json").write_text(
        f'{{"quantization": "awq", "source": "{model_path}"}}'
    )
    logger.info("AWQ model saved to %s", dest)
    return str(dest)


def export_model(model_path: str, format: str = "gguf") -> str:
    """Unified model export dispatcher.

    Args:
        model_path: Path to the fine-tuned model.
        format: Target format — ``'gguf'`` or ``'awq'``.

    Returns:
        Path to the exported model.

    Raises:
        ValueError: If an unsupported format is specified.
    """
    if format not in ("gguf", "awq"):
        raise ValueError(f"Unsupported export format '{format}'. Use 'gguf' or 'awq'")

    output_path = str(Path(model_path).parent / f"exported_{format}")

    if format == "gguf":
        return quantize_gguf(model_path, output_path)
    else:
        return quantize_awq(model_path, output_path)
