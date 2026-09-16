from uuid import UUID

import pytest
from pydantic import ValidationError

from rag_eval.models import EvalReport, Sample, SampleScore


def test_sample_valid_complete() -> None:
    data = {
        "question": "What is Python?",
        "answer": "Python is a programming language.",
        "context": ["Python is an interpreted, high-level programming language."],
        "golden_documents": ["Python is an interpreted, high-level programming language."],
        "retrieved": ["Python is an interpreted, high-level programming language."],
        "id": "custom-id-1",
    }
    sample = Sample.model_validate(data)
    assert sample.question == "What is Python?"
    assert sample.answer == "Python is a programming language."
    assert sample.context == ["Python is an interpreted, high-level programming language."]
    assert sample.golden_documents == ["Python is an interpreted, high-level programming language."]
    assert sample.retrieved == ["Python is an interpreted, high-level programming language."]
    assert sample.id == "custom-id-1"


def test_sample_missing_question_raises_validation_error() -> None:
    data = {
        "answer": "Python is a programming language.",
        "context": ["Python is an interpreted, high-level programming language."],
    }
    with pytest.raises(ValidationError):
        Sample.model_validate(data)


def test_sample_empty_context_raises_validation_error() -> None:
    data = {
        "question": "What is Python?",
        "answer": "Python is a programming language.",
        "context": [],
    }
    with pytest.raises(ValidationError):
        Sample.model_validate(data)


def test_sample_defaults() -> None:
    data = {
        "question": "What is RAG?",
        "answer": "Retrieval-Augmented Generation.",
        "context": ["RAG combines retrieval with LLM generation."],
    }
    sample = Sample.model_validate(data)
    assert sample.retrieved is None
    assert sample.golden_documents is None
    # ID should be a valid UUID4 string
    assert sample.id is not None
    assert UUID(sample.id).version == 4


def test_sample_score_and_eval_report() -> None:
    score = SampleScore(
        sample_id="sample-1",
        retrieval_precision_at_k=1.0,
        faithfulness=0.8,
        faithfulness_method="claim_overlap",
        claims=["Claim 1", "Claim 2"],
        unsupported_claims=["Claim 2"],
    )
    report = EvalReport(
        n_samples=1,
        mean_retrieval_precision_at_k=1.0,
        mean_faithfulness=0.8,
        per_sample=[score],
    )
    assert report.n_samples == 1
    assert report.mean_retrieval_precision_at_k == 1.0
    assert report.mean_faithfulness == 0.8
    assert len(report.per_sample) == 1
    assert report.per_sample[0].sample_id == "sample-1"
