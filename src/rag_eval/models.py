"""Data models for rag-eval."""

from uuid import uuid4

from pydantic import BaseModel, Field


class Sample(BaseModel):
    """An evaluation sample representing a single query, generation, and retrieval context."""

    question: str
    answer: str
    context: list[str] = Field(..., min_length=1)
    golden_documents: list[str] | None = None
    retrieved: list[str] | None = None
    id: str = Field(default_factory=lambda: str(uuid4()))


class SampleScore(BaseModel):
    """Evaluation scores and extracted claim diagnostics for a single sample."""

    sample_id: str
    retrieval_precision_at_k: float | None = None
    faithfulness: float | None = None
    faithfulness_method: str
    claims: list[str] | None = None
    unsupported_claims: list[str] | None = None


class EvalReport(BaseModel):
    """Aggregated evaluation report across all evaluated samples."""

    n_samples: int
    mean_retrieval_precision_at_k: float | None = None
    mean_faithfulness: float | None = None
    per_sample: list[SampleScore]
