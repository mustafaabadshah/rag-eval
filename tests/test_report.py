from rich.console import Console
from rich.table import Table

from rag_eval.models import EvalReport, SampleScore
from rag_eval.report import build_report, render_json, render_table


def test_build_report_computes_correct_means() -> None:
    scores = [
        SampleScore(
            sample_id="s1",
            retrieval_precision_at_k=1.0,
            faithfulness=1.0,
            faithfulness_method="claim_overlap",
        ),
        SampleScore(
            sample_id="s2",
            retrieval_precision_at_k=0.0,
            faithfulness=0.5,
            faithfulness_method="claim_overlap",
        ),
        SampleScore(
            sample_id="s3",
            retrieval_precision_at_k=None,
            faithfulness=0.6,
            faithfulness_method="claim_overlap",
        ),
    ]

    report = build_report(scores)
    assert report.n_samples == 3
    assert report.mean_retrieval_precision_at_k == 0.5
    assert report.mean_faithfulness == pytest.approx(0.7)
    assert len(report.per_sample) == 3


def test_render_json_roundtrip() -> None:
    scores = [
        SampleScore(
            sample_id="s1",
            retrieval_precision_at_k=0.8,
            faithfulness=0.9,
            faithfulness_method="claim_overlap",
        )
    ]
    report = build_report(scores)
    json_str = render_json(report)

    # Validate JSON round-trip
    restored = EvalReport.model_validate_json(json_str)
    assert restored.n_samples == report.n_samples
    assert restored.mean_faithfulness == report.mean_faithfulness
    assert restored.mean_retrieval_precision_at_k == report.mean_retrieval_precision_at_k


def test_render_table() -> None:
    scores = [
        SampleScore(
            sample_id="s1",
            retrieval_precision_at_k=1.0,
            faithfulness=0.75,
            faithfulness_method="claim_overlap",
        )
    ]
    report = build_report(scores)
    table = render_table(report)

    assert isinstance(table, Table)

    console = Console(record=True, width=120)
    console.print(table)
    output = console.export_text()

    assert "faithfulness" in output.lower()
    assert "0.75" in output or "75.0%" in output


import pytest  # noqa: E402
