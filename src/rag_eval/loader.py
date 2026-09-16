"""JSONL loader and validator for rag-eval."""

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from rag_eval.models import Sample

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """Raised when an error occurs while loading or parsing a JSONL file."""


def load(path: str | Path, *, skip_errors: bool = False) -> list[Sample]:
    """Load and validate evaluation samples from a JSONL file.

    Args:
        path: Path to the JSONL file.
        skip_errors: If True, log warnings for invalid lines and continue loading.
            If False, raise LoaderError on the first invalid line.

    Returns:
        List of validated Sample instances.

    Raises:
        LoaderError: If file cannot be read, or if an invalid line is encountered
            when skip_errors is False.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise LoaderError(f"File not found: {file_path}")

    samples: list[Sample] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                raw_data = json.loads(stripped)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON syntax at line {line_num}: {e}"
                if skip_errors:
                    logger.warning(msg)
                    continue
                raise LoaderError(msg) from e

            if not isinstance(raw_data, dict):
                msg = f"Invalid data format at line {line_num}: expected JSON object"
                if skip_errors:
                    logger.warning(msg)
                    continue
                raise LoaderError(msg)

            try:
                sample = Sample.model_validate(raw_data)
                samples.append(sample)
            except ValidationError as e:
                msg = f"Validation error at line {line_num}: {e}"
                if skip_errors:
                    logger.warning(msg)
                    continue
                raise LoaderError(msg) from e

    return samples
