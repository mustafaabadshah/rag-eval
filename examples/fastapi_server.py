"""FastAPI evaluation server for rag-eval."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag_eval.faithfulness import faithfulness
from rag_eval.models import EvalReport, Sample, SampleScore
from rag_eval.report import build_report
from rag_eval.retrieval import precision_at_k

app = FastAPI(
    title="rag-eval API",
    description="Evaluation API for RAG pipelines: retrieval precision and answer faithfulness.",
    version="0.1.0",
)


class EvaluateRequest(BaseModel):
    samples: list[Sample]
    k: int = Field(default=5, ge=1)
    claims_only: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvalReport)
def evaluate_endpoint(request: EvaluateRequest) -> EvalReport:
    """Evaluate a batch of samples on precision@k and answer faithfulness."""
    scores: list[SampleScore] = []
    for sample in request.samples:
        retrieval_score: float | None = None
        if not request.claims_only and sample.retrieved is not None and sample.golden_documents is not None:
            retrieval_score = precision_at_k(sample.retrieved, sample.golden_documents, k=request.k)

        faith_score, supported, unsupported = faithfulness(sample.answer, sample.context)

        all_claims = supported + unsupported
        scores.append(
            SampleScore(
                sample_id=sample.id,
                retrieval_precision_at_k=retrieval_score,
                faithfulness=faith_score,
                faithfulness_method="claim_overlap",
                claims=all_claims,
                unsupported_claims=unsupported,
            )
        )

    return build_report(scores)
