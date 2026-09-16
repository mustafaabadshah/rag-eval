import logging
from pathlib import Path

import pytest

from rag_eval.loader import LoaderError, load
from rag_eval.models import Sample

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_JSONL = FIXTURES_DIR / "sample.jsonl"


def test_load_strict_mode_raises_loader_error() -> None:
    with pytest.raises(LoaderError) as exc_info:
        load(SAMPLE_JSONL)
    # The first invalid line in sample.jsonl is line 3
    assert "line 3" in str(exc_info.value).lower()


def test_load_skip_errors_returns_three_samples(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        samples = load(SAMPLE_JSONL, skip_errors=True)
    assert len(samples) == 3
    assert all(isinstance(s, Sample) for s in samples)
    assert [s.id for s in samples] == ["sample-1", "sample-2", "sample-4"]
    # Check that warnings were logged for invalid lines
    assert any("line 3" in record.message.lower() for record in caplog.records)
    assert any("line 5" in record.message.lower() for record in caplog.records)


def test_load_valid_file(tmp_path: Path) -> None:
    valid_file = tmp_path / "valid.jsonl"
    valid_file.write_text(
        '{"question": "Q1", "answer": "A1", "context": ["C1"]}\n'
        '{"question": "Q2", "answer": "A2", "context": ["C2"]}\n',
        encoding="utf-8",
    )
    samples = load(valid_file)
    assert len(samples) == 2
    assert samples[0].question == "Q1"
    assert samples[1].question == "Q2"


def test_load_invalid_json_syntax(tmp_path: Path) -> None:
    broken_file = tmp_path / "broken.jsonl"
    broken_file.write_text("{not valid json\n", encoding="utf-8")
    with pytest.raises(LoaderError) as exc_info:
        load(broken_file)
    assert "line 1" in str(exc_info.value).lower()
