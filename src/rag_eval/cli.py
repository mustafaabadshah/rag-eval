"""Command-line interface for rag-eval."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from rag_eval.faithfulness import faithfulness
from rag_eval.loader import LoaderError, load
from rag_eval.models import SampleScore
from rag_eval.report import build_report, render_json, render_table
from rag_eval.retrieval import precision_at_k

app = typer.Typer(
    name="rag-eval",
    help="Score any RAG pipeline: retrieval precision + answer faithfulness.",
    no_args_is_help=True,
)
err_console = Console(stderr=True)


@app.callback()
def main() -> None:
    """Score any RAG pipeline on retrieval precision and answer faithfulness."""


@app.command(name="eval")
def evaluate(
    data: Annotated[Path, typer.Argument(help="Path to input JSONL evaluation data file")],
    output_json: Annotated[bool, typer.Option("--json", help="Output results in JSON format")] = False,
    skip_errors: Annotated[bool, typer.Option("--skip-errors", help="Skip invalid lines instead of failing")] = False,
    k: Annotated[int, typer.Option("--k", min=1, help="Rank cutoff k for precision@k retrieval metric")] = 5,
    claims_only: Annotated[
        bool, typer.Option("--claims-only", help="Skip retrieval metrics and evaluate answer faithfulness only")
    ] = False,
) -> None:
    """Evaluate a JSONL dataset on retrieval precision and answer faithfulness."""
    if not data.exists():
        err_console.print(f"[bold red]Error:[/bold red] File not found: {data}")
        raise typer.Exit(code=1)

    try:
        samples = load(data, skip_errors=skip_errors)
    except LoaderError as err:
        err_console.print(f"[bold red]Error:[/bold red] {err}")
        raise typer.Exit(code=1) from err

    scores: list[SampleScore] = []
    for sample in samples:
        retrieval_score: float | None = None
        if not claims_only and sample.retrieved is not None and sample.golden_documents is not None:
            retrieval_score = precision_at_k(sample.retrieved, sample.golden_documents, k=k)

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

    report = build_report(scores)

    if output_json:
        typer.echo(render_json(report))
    else:
        out_console = Console()
        out_console.print(render_table(report))


if __name__ == "__main__":
    app()
