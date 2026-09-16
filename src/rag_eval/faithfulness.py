"""Answer faithfulness scoring using claim decomposition and context overlap."""

import re

SUPPORT_THRESHOLD: float = 0.8

STOPWORDS: set[str] = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "it",
    "this",
}


def extract_claims(answer: str) -> list[str]:
    """Extract atomic claims from an answer string using heuristic splitting.

    Splits sentences on terminal punctuation ([.!?]), and further subdivides
    clauses on semicolons and contrasting coordinating conjunctions (', but ', etc.).

    Args:
        answer: The generated answer string.

    Returns:
        List of non-empty extracted claim strings.
    """
    if not answer or not answer.strip():
        return []

    # First split into sentences on [.!?]
    sentences = re.split(r"[.!?]+", answer)

    claims: list[str] = []
    # Further split clauses on semicolons or ", but ", ", however ", etc.
    clause_delimiter = re.compile(r";\s*|,\s+(?:but|however|whereas|although)\s+", re.IGNORECASE)

    for sentence in sentences:
        sub_clauses = clause_delimiter.split(sentence)
        for clause in sub_clauses:
            cleaned = clause.strip()
            if cleaned:
                claims.append(cleaned)

    return claims


def claim_supported(claim: str, context: list[str]) -> bool:
    """Determine whether a claim is supported by the retrieved context.

    Tokenizes the claim into alphanumeric content words (filtering out stopwords).
    A claim is supported if >= SUPPORT_THRESHOLD (80%) of its content words are present
    in the union of context content words.

    Args:
        claim: The claim string to verify.
        context: List of context document/chunk strings.

    Returns:
        True if the claim is supported by the context, False otherwise.
    """
    claim_tokens = re.findall(r"[a-z0-9]+", claim.lower())
    content_words = [token for token in claim_tokens if token not in STOPWORDS]

    # If the claim contains no content words, consider it supported by default
    if not content_words:
        return True

    context_text = " ".join(context).lower()
    context_tokens = set(re.findall(r"[a-z0-9]+", context_text))

    overlap = sum(1 for word in content_words if word in context_tokens)
    ratio = overlap / len(content_words)
    return ratio >= SUPPORT_THRESHOLD


def faithfulness(answer: str, context: list[str]) -> tuple[float, list[str], list[str]]:
    """Compute faithfulness score for an answer given retrieved context chunks.

    Args:
        answer: The generated answer text.
        context: List of context chunk strings.

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
        if claim_supported(claim, context):
            supported.append(claim)
        else:
            unsupported.append(claim)

    score = len(supported) / len(claims)
    return score, supported, unsupported
