import json
from pathlib import Path

from typer.testing import CliRunner

from rag_eval.cli import app

runner = CliRunner()
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_JSONL = FIXTURES_DIR / "sample.jsonl"


def test_cli_eval_table_output(tmp_path: Path) -> None:
    valid_file = tmp_path / "valid.jsonl"
    valid_file.write_text(
        '{"question": "Q1", "answer": "Paris is capital.", "context": ["Paris is capital of France."]}\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(valid_file)])
    assert result.exit_code == 0
    assert "faithfulness" in result.stdout.lower()


def test_cli_eval_skip_errors_on_bad_lines() -> None:
    result = runner.invoke(app, ["eval", str(SAMPLE_JSONL), "--skip-errors"])
    assert result.exit_code == 0
    assert "faithfulness" in result.stdout.lower()


def test_cli_eval_json_flag() -> None:
    result = runner.invoke(app, ["eval", str(SAMPLE_JSONL), "--skip-errors", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert "mean_faithfulness" in parsed
    assert parsed["n_samples"] == 3


def test_cli_eval_strict_mode_fails_on_bad_file() -> None:
    result = runner.invoke(app, ["eval", str(SAMPLE_JSONL)])
    assert result.exit_code == 1
    assert "validation error" in result.output.lower() or "line 3" in result.output.lower()


def test_cli_eval_missing_file() -> None:
    result = runner.invoke(app, ["eval", "non_existent_file.jsonl"])
    assert result.exit_code == 1
    # Error message should be non-empty and have no Python traceback
    assert "not found" in result.output.lower() or "error" in result.output.lower()
    assert "Traceback" not in result.output


def test_cli_eval_claims_only(tmp_path: Path) -> None:
    valid_file = tmp_path / "valid.jsonl"
    valid_file.write_text(
        '{"question": "Q1", "answer": "Paris is capital.", "context": ["Paris is capital of France."], '
        '"golden_documents": ["Paris is capital of France."], "retrieved": ["Paris is capital of France."]}\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(valid_file), "--claims-only", "--json"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["mean_retrieval_precision_at_k"] is None


def test_cli_eval_unsupported_judge(tmp_path: Path) -> None:
    valid_file = tmp_path / "valid.jsonl"
    valid_file.write_text(
        '{"question": "Q1", "answer": "Paris is capital.", "context": ["Paris is capital of France."]}\n',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", str(valid_file), "--judge", "anthropic"])
    assert result.exit_code == 1
    assert "unsupported judge" in result.output.lower()

