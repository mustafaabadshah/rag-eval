"""Retrieval metrics calculation for rag-eval."""


def precision_at_k(retrieved: list[str] | None, golden: list[str] | None, k: int = 5) -> float | None:
    """Calculate retrieval precision@k with set semantics.

    Precision@k is defined as:
        |set(retrieved[:k]) ∩ set(golden)| / k

    Args:
        retrieved: List of chunk strings retrieved by the retriever in ranked order.
        golden: List of ground-truth relevant chunk strings.
        k: The rank cutoff (positive integer).

    Returns:
        Score between 0.0 and 1.0, or None if retrieved or golden is None.
    """
    if retrieved is None or golden is None:
        return None

    if k <= 0:
        return 0.0

    retrieved_top_k = retrieved[:k]
    matches = set(retrieved_top_k) & set(golden)
    return len(matches) / float(k)


def mean_precision_at_k(scores: list[float | None]) -> float | None:
    """Calculate arithmetic mean over all non-None precision scores.

    Args:
        scores: Sequence of sample precision scores (or None when omitted).

    Returns:
        Mean precision score as a float, or None if all scores are None or empty.
    """
    valid_scores = [s for s in scores if s is not None]
    if not valid_scores:
        return None
    return sum(valid_scores) / len(valid_scores)
