import pytest

from rag_eval.faithfulness import (
    SUPPORT_THRESHOLD,
    claim_supported,
    extract_claims,
    faithfulness,
)


def test_extract_claims() -> None:
    answer = "Paris is the capital of France. It is beautiful; tourists love it, but it rains often!"
    claims = extract_claims(answer)
    assert len(claims) >= 3
    assert any("Paris is the capital of France" in c for c in claims)


def test_claim_supported_directly() -> None:
    context = ["The sun is a star at the center of the Solar System."]
    assert claim_supported("The sun is a star.", context) is True
    assert claim_supported("The moon is made of green cheese.", context) is False
    # Only stopwords
    assert claim_supported("It is this and it.", context) is True


def test_faithfulness_fully_supported() -> None:
    context = [
        "The Eiffel Tower is located in Paris, France. It was completed in 1889."
    ]
    answer = "The Eiffel Tower is located in Paris. It was completed in 1889."
    score, supported, unsupported = faithfulness(answer, context)
    assert score == 1.0
    assert len(supported) == 2
    assert len(unsupported) == 0


def test_faithfulness_one_unsupported_out_of_four() -> None:
    context = [
        "Mars is the fourth planet from the Sun. It has a thin atmosphere. It has two moons named Phobos and Deimos."
    ]
    answer = (
        "Mars is the fourth planet from the Sun. "
        "It has a thin atmosphere. "
        "It has two moons. "
        "Humans have landed on Mars."
    )
    score, supported, unsupported = faithfulness(answer, context)
    assert score == pytest.approx(0.75)
    assert len(supported) == 3
    assert len(unsupported) == 1
    assert "Humans have landed on Mars" in unsupported[0]


def test_faithfulness_empty_answer() -> None:
    context = ["Some context document."]
    score, supported, unsupported = faithfulness("", context)
    assert score == 1.0
    assert supported == []
    assert unsupported == []


def test_faithfulness_case_and_punctuation_insensitivity() -> None:
    context = ["Artificial Intelligence (AI) revolutionized modern computing!"]
    answer = "artificial intelligence revolutionized modern computing."
    score, supported, unsupported = faithfulness(answer, context)
    assert score == 1.0
    assert len(supported) == 1
    assert len(unsupported) == 0


def test_support_threshold_constant() -> None:
    assert SUPPORT_THRESHOLD == 0.8


class FakeJudge:
    def __init__(self, supported_claims: set[str]) -> None:
        self.supported_claims = supported_claims

    def verify(self, claim: str, context: list[str]) -> bool:
        return claim in self.supported_claims


def test_faithfulness_with_judge() -> None:
    from rag_eval.judge import faithfulness_with_judge

    context = ["Alpha Centauri is a star system."]
    answer = "Alpha Centauri is a star system. It has 10 planets."
    fake_judge = FakeJudge(supported_claims={"Alpha Centauri is a star system"})

    score, supported, unsupported = faithfulness_with_judge(answer, context, fake_judge)
    assert score == 0.5
    assert supported == ["Alpha Centauri is a star system"]
    assert unsupported == ["It has 10 planets"]


def test_openai_judge_missing_extra_raises_runtime_error() -> None:
    import importlib.util

    from rag_eval.judge import OpenAIJudge

    # If openai is not installed in the default test environment, it should raise RuntimeError
    if importlib.util.find_spec("openai") is None:
        with pytest.raises(RuntimeError) as exc_info:
            OpenAIJudge()
        assert "pip install rag-eval[judge]" in str(exc_info.value)

