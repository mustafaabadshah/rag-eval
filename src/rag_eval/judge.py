"""Optional LLM-based judge for claim verification."""

from typing import Any, Protocol, runtime_checkable

from rag_eval.faithfulness import extract_claims


@runtime_checkable
class Judge(Protocol):
    """Protocol for pluggable verification judges."""

    def verify(self, claim: str, context: list[str]) -> bool:
        """Verify whether a single claim is supported by the context."""
        ...


class OpenAIJudge:
    """LLM judge implementation using OpenAI API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        import importlib

        try:
            openai = importlib.import_module("openai")
        except ImportError as err:
            raise RuntimeError(
                "OpenAI extra not installed. Install with: pip install rag-eval[judge]"
            ) from err

        self.model = model
        self.client: Any = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def verify(self, claim: str, context: list[str]) -> bool:
        """Verify whether a claim is supported by the context using an LLM prompt.

        Prompt: 'Answer strictly yes or no: is the claim supported by the context?'
        """
        prompt = (
            "Answer strictly yes or no: is the claim supported by the context?\n\n"
            f"Claim: {claim}\n\n"
            f"Context:\n{' '.join(context)}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        first_choice = response.choices[0]
        content = first_choice.message.content or ""
        return content.strip().lower().startswith("yes")


def faithfulness_with_judge(
    answer: str, context: list[str], judge: Judge
) -> tuple[float, list[str], list[str]]:
    """Compute faithfulness score by verifying claims using an external Judge.

    Args:
        answer: The generated answer string.
        context: List of context document/chunk strings.
        judge: Pluggable Judge instance implementing the verify method.

    Returns:
        A tuple of:
            - faithfulness score (float between 0.0 and 1.0; 1.0 if no claims)
            - list of supported claim strings
            - list of unsupported claim strings
    """
    claims = extract_claims(answer)
    if not claims:
        return 1.0, [], []

    supported: list[str] = []
    unsupported: list[str] = []

    for claim in claims:
        if judge.verify(claim, context):
            supported.append(claim)
        else:
            unsupported.append(claim)

    score = len(supported) / len(claims)
    return score, supported, unsupported
