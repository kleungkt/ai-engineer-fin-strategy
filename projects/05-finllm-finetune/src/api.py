"""FastAPI service for financial LLM fine-tuning pipeline."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .trainer import TrainingConfig

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Financial LLM Fine-tune API",
    description="Train, evaluate, and serve fine-tuned financial language models.",
    version="0.1.0",
)

# In-memory job store (production would use Redis / database)
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    """Training job request."""

    config: TrainingConfig = Field(default_factory=TrainingConfig)
    data_path: str = Field(description="Path to training JSONL file")


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    result: Any = None


class EvaluateRequest(BaseModel):
    model_path: str
    test_data_path: str


class EvaluateResponse(BaseModel):
    results: dict[str, Any]


class GenerateRequest(BaseModel):
    model_path: str
    prompt: str
    max_new_tokens: int = 256


class GenerateResponse(BaseModel):
    generated_text: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/train", response_model=TrainResponse)
async def start_training(request: TrainRequest):
    """Start a LoRA fine-tuning job (background)."""
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    _jobs[job_id] = {
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "config": request.config.model_dump(),
        "data_path": request.data_path,
        "result": None,
    }

    # In production this would launch an async background task or submit to a queue.
    # For the API skeleton we just mark it as pending.
    logger.info("Training job %s created", job_id)

    return TrainResponse(
        job_id=job_id,
        status="pending",
        message="Training job submitted. Poll /status/{job_id} for progress.",
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str):
    """Check the status of a training job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        result=job.get("result"),
    )


@app.post("/evaluate", response_model=EvaluateResponse)
async def run_evaluation(request: EvaluateRequest):
    """Run evaluation benchmarks on a fine-tuned model."""
    from .data_pipeline import load_raw_data
    from .evaluator import run_benchmark

    try:
        test_data = load_raw_data(request.test_data_path)
        test_dicts = [s.model_dump() for s in test_data]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        results = run_benchmark(request.model_path, test_dicts)
        serialised = {k: v.model_dump() for k, v in results.items()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}")

    return EvaluateResponse(results=serialised)


@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using a fine-tuned model."""
    from .evaluator import evaluate_generation

    try:
        outputs = evaluate_generation(request.model_path, [request.prompt])
        return GenerateResponse(generated_text=outputs[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")
