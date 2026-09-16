"""Report generation and visualization for rag-eval."""

from rich.box import ROUNDED
from rich.table import Table

from rag_eval.models import EvalReport, SampleScore


def build_report(scores: list[SampleScore]) -> EvalReport:
    """Aggregate individual sample scores into an evaluation report.

    Args:
        scores: Sequence of SampleScore items.

    Returns:
        EvalReport containing sample counts, arithmetic means, and individual scores.
    """
    n_samples = len(scores)
    retrieval_scores = [s.retrieval_precision_at_k for s in scores if s.retrieval_precision_at_k is not None]
    faithfulness_scores = [s.faithfulness for s in scores if s.faithfulness is not None]

    mean_retrieval = (sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else None
    mean_faith = (sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else None

    return EvalReport(
        n_samples=n_samples,
        mean_retrieval_precision_at_k=mean_retrieval,
        mean_faithfulness=mean_faith,
        per_sample=scores,
    )


def render_json(report: EvalReport) -> str:
    """Render an EvalReport as a formatted JSON string.

    Args:
        report: The evaluation report to serialize.

    Returns:
        Pretty-printed JSON representation.
    """
    return report.model_dump_json(indent=2)


def render_table(report: EvalReport) -> Table:
    """Render an EvalReport as a styled rich Table.

    Args:
        report: The evaluation report to format.

    Returns:
        A rich Table displaying evaluation metrics, sample counts, and methods.
    """
    table = Table(
        title="[bold cyan]RAG-EVAL Evaluation Summary[/bold cyan]",
        box=ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )

    table.add_column("Metric", style="bold", justify="left")
    table.add_column("Score / Value", justify="right", style="green")
    table.add_column("Method & Details", justify="left", style="dim")

    table.add_row(
        "Evaluated Samples",
        str(report.n_samples),
        "Total processed samples",
    )

    retrieval_display = (
        f"{report.mean_retrieval_precision_at_k:.4f}"
        if report.mean_retrieval_precision_at_k is not None
        else "N/A (skipped or no golden docs)"
    )
    table.add_row(
        "Mean Retrieval Precision@k",
        retrieval_display,
        "Precision at rank cutoff k with set semantics",
    )

    methods = {s.faithfulness_method for s in report.per_sample}
    method_desc = ", ".join(sorted(methods)) if methods else "claim_overlap"

    faith_display = (
        f"{report.mean_faithfulness:.4f}"
        if report.mean_faithfulness is not None
        else "N/A"
    )
    table.add_row(
        "Mean Faithfulness",
        faith_display,
        f"Answer verification via {method_desc}",
    )

    return table
