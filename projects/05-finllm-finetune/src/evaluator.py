"""Evaluate fine-tuned financial LLM quality."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EvalResult(BaseModel):
    """Single evaluation metric result."""

    metric_name: str
    score: float
    details: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str):
    """Lazily load a fine-tuned model and tokenizer.

    Raises ImportError if the required libraries are not installed.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            f"Install transformers and peft for evaluation: {exc}"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = PeftModel.from_pretrained(model, model_path)
    except Exception:
        pass  # Base model already loaded or merged
    model.eval()
    return model, tokenizer


def _generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response for a single prompt."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the output
    return generated[len(prompt) :].strip()


def _simple_bleu(reference: str, hypothesis: str) -> float:
    """Compute a simplified BLEU-1 score (unigram precision)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)
    clipped = sum(min(hyp_counts[t], ref_counts.get(t, 0)) for t in hyp_counts)
    precision = clipped / len(hyp_tokens)
    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1)) if ref_tokens else 0.0
    return bp * precision


def _simple_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L (longest common subsequence F1)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    recall = lcs_len / m
    precision = lcs_len / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------


def evaluate_sentiment(model_path: str, test_data: list) -> EvalResult:
    """Evaluate sentiment classification accuracy.

    Args:
        model_path: Path to the fine-tuned model or adapter directory.
        test_data: List of dicts with ``input`` and ``output`` keys where
            ``output`` contains the expected sentiment label.

    Returns:
        EvalResult with accuracy metric.
    """
    model, tokenizer = _load_model(model_path)
    correct = 0
    total = len(test_data)
    details: dict[str, int] = {"correct": 0, "total": total}

    for sample in test_data:
        prompt = (
            "Analyze the sentiment of the following financial news headline.\n"
            f"{sample['input']}\nSentiment:"
        )
        prediction = _generate_text(model, tokenizer, prompt, max_new_tokens=10)
        expected = sample["output"].lower().strip()
        # Check if expected label appears in prediction
        if any(label in prediction.lower() for label in ["bullish", "bearish", "neutral"]):
            pred_label = next(
                label for label in ["bullish", "bearish", "neutral"]
                if label in prediction.lower()
            )
            if pred_label in expected:
                correct += 1

    details["correct"] = correct
    score = correct / total if total > 0 else 0.0
    return EvalResult(metric_name="sentiment_accuracy", score=score, details=details)


def evaluate_qa(model_path: str, test_data: list) -> EvalResult:
    """Evaluate question-answering quality using BLEU and ROUGE.

    Args:
        model_path: Path to the fine-tuned model.
        test_data: List of dicts with ``input`` (question) and ``output`` (answer).

    Returns:
        EvalResult with BLEU and ROUGE scores.
    """
    model, tokenizer = _load_model(model_path)
    bleu_scores: list[float] = []
    rouge_scores: list[float] = []

    for sample in test_data:
        prompt = (
            "Answer the following financial question accurately and concisely.\n"
            f"Question: {sample['input']}\nAnswer:"
        )
        prediction = _generate_text(model, tokenizer, prompt)
        expected = sample["output"]
        bleu_scores.append(_simple_bleu(expected, prediction))
        rouge_scores.append(_simple_rouge_l(expected, prediction))

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    composite = (avg_bleu + avg_rouge) / 2

    return EvalResult(
        metric_name="qa_quality",
        score=composite,
        details={"bleu": avg_bleu, "rouge_l": avg_rouge, "num_samples": len(test_data)},
    )


def evaluate_generation(model_path: str, prompts: list[str]) -> list[str]:
    """Generate model outputs for a list of prompts.

    Args:
        model_path: Path to the fine-tuned model.
        prompts: List of prompt strings.

    Returns:
        List of generated text strings.
    """
    model, tokenizer = _load_model(model_path)
    return [_generate_text(model, tokenizer, p) for p in prompts]


def run_benchmark(model_path: str, test_data: list) -> dict[str, EvalResult]:
    """Run all evaluation benchmarks on the model.

    Args:
        model_path: Path to the fine-tuned model.
        test_data: Combined test dataset with ``category`` field.

    Returns:
        Dictionary mapping metric name to EvalResult.
    """
    results: dict[str, EvalResult] = {}

    # Split by category
    sentiment_data = [d for d in test_data if d.get("category") == "sentiment"]
    qa_data = [d for d in test_data if d.get("category") == "qa"]

    if sentiment_data:
        results["sentiment_accuracy"] = evaluate_sentiment(model_path, sentiment_data)
    if qa_data:
        results["qa_quality"] = evaluate_qa(model_path, qa_data)

    # Generation quality on a sample of prompts
    sample_prompts = [d.get("input", "") for d in test_data[:5]]
    if sample_prompts:
        generations = evaluate_generation(model_path, sample_prompts)
        results["generation_count"] = EvalResult(
            metric_name="generation_count",
            score=float(len(generations)),
            details={"num_generated": len(generations)},
        )

    return results
