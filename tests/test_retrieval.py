from rag_eval.retrieval import mean_precision_at_k, precision_at_k


def test_precision_at_k_exact_match_rank_1() -> None:
    retrieved = ["doc1", "doc2"]
    golden = ["doc1"]
    score = precision_at_k(retrieved, golden, k=1)
    assert score == 1.0


def test_precision_at_k_rank_3_of_5() -> None:
    retrieved = ["docA", "docB", "docC", "docD", "docE"]
    golden = ["docC"]
    score = precision_at_k(retrieved, golden, k=5)
    assert score == pytest.approx(0.2)


def test_precision_at_k_no_overlap() -> None:
    retrieved = ["docA", "docB"]
    golden = ["docC"]
    score = precision_at_k(retrieved, golden, k=2)
    assert score == 0.0


def test_precision_at_k_none_inputs() -> None:
    assert precision_at_k(None, ["docA"], k=5) is None
    assert precision_at_k(["docA"], None, k=5) is None
    assert precision_at_k(None, None, k=5) is None


def test_precision_at_k_duplicate_golden_docs_set_semantics() -> None:
    retrieved = ["doc1", "doc2"]
    # Duplicates in golden documents should not inflate the match count
    golden = ["doc1", "doc1"]
    score = precision_at_k(retrieved, golden, k=2)
    assert score == 0.5


def test_precision_at_k_k_zero_or_negative() -> None:
    assert precision_at_k(["doc1"], ["doc1"], k=0) == 0.0


def test_mean_precision_at_k() -> None:
    assert mean_precision_at_k([1.0, 0.2, 0.0]) == pytest.approx(0.4)
    assert mean_precision_at_k([1.0, None, 0.0]) == pytest.approx(0.5)
    assert mean_precision_at_k([None, None]) is None
    assert mean_precision_at_k([]) is None


import pytest  # noqa: E402
