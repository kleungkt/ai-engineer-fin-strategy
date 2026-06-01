"""CLI entry-point for the Financial LLM fine-tuning pipeline.

Usage:
    python -m src.main generate --n 200 --output data/train.jsonl
    python -m src.main train   --config config.json --data data/train.jsonl
    python -m src.main evaluate --model output/final_adapter --test data/test.jsonl
    python -m src.main export  --model output/final_adapter --format gguf
    python -m src.main serve   --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("finllm")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a synthetic training dataset."""
    from .data_pipeline import generate_sample_dataset, save_dataset

    logger.info("Generating %d synthetic financial samples …", args.n)
    samples = generate_sample_dataset(args.n)
    save_dataset(samples, args.output)
    logger.info("Saved %d samples to %s", len(samples), args.output)


def cmd_train(args: argparse.Namespace) -> None:
    """Start LoRA fine-tuning."""
    from .data_pipeline import load_raw_data, format_for_training, split_dataset
    from .trainer import TrainingConfig, train

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            cfg_dict = json.load(f)
        config = TrainingConfig(**cfg_dict)
    else:
        config = TrainingConfig()

    # Load data
    logger.info("Loading training data from %s …", args.data)
    samples = load_raw_data(args.data)
    formatted = format_for_training(samples, format="alpaca")
    train_data, eval_data = split_dataset(formatted, test_ratio=0.1)

    logger.info("Starting training: %d train / %d eval samples", len(train_data), len(eval_data))
    output_path = train(config, train_data, eval_data)
    logger.info("Training complete. Adapters saved to %s", output_path)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run model evaluation."""
    from .data_pipeline import load_raw_data
    from .evaluator import run_benchmark

    logger.info("Loading test data from %s …", args.test)
    test_samples = load_raw_data(args.test)
    test_dicts = [s.model_dump() for s in test_samples]

    logger.info("Running benchmark on model at %s …", args.model)
    results = run_benchmark(args.model, test_dicts)

    for name, result in results.items():
        logger.info("  %s: %.4f  (details: %s)", name, result.score, result.details)


def cmd_export(args: argparse.Namespace) -> None:
    """Export / quantize model for deployment."""
    from .quantize import export_model

    logger.info("Exporting model %s to format=%s …", args.model, args.format)
    output_path = export_model(args.model, args.format)
    logger.info("Model exported to %s", output_path)


def cmd_serve(args: argparse.Namespace) -> None:
    """Launch the FastAPI server."""
    import uvicorn

    logger.info("Starting API server on %s:%d …", args.host, args.port)
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="finllm",
        description="Financial LLM Fine-tuning Pipeline",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    gen_p = sub.add_parser("generate", help="Generate synthetic training data")
    gen_p.add_argument("--n", type=int, default=100, help="Number of samples")
    gen_p.add_argument("--output", type=str, default="data/train.jsonl", help="Output path")

    # train
    trn_p = sub.add_parser("train", help="Start LoRA fine-tuning")
    trn_p.add_argument("--config", type=str, default=None, help="JSON config file")
    trn_p.add_argument("--data", type=str, required=True, help="Training data JSONL")

    # evaluate
    evl_p = sub.add_parser("evaluate", help="Evaluate fine-tuned model")
    evl_p.add_argument("--model", type=str, required=True, help="Model/adapter path")
    evl_p.add_argument("--test", type=str, required=True, help="Test data JSONL")

    # export
    exp_p = sub.add_parser("export", help="Quantize and export model")
    exp_p.add_argument("--model", type=str, required=True, help="Model path")
    exp_p.add_argument("--format", type=str, default="gguf", choices=["gguf", "awq"])

    # serve
    srv_p = sub.add_parser("serve", help="Start FastAPI server")
    srv_p.add_argument("--host", type=str, default="0.0.0.0")
    srv_p.add_argument("--port", type=int, default=8000)
    srv_p.add_argument("--reload", action="store_true", default=False)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "serve": cmd_serve,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
