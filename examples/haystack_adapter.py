"""Haystack 2.x pipeline output adapter for rag-eval.

Example Haystack 2.0 pipeline snippet producing the evaluation input:

```python
from haystack import Document, Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

# 1. Initialize Document Store & Retriever
document_store = InMemoryDocumentStore()
document_store.write_documents(
    [Document(content="Paris is the capital and largest city of France.")]
)
retriever = InMemoryBM25Retriever(document_store=document_store)

# 2. Build Pipeline
prompt_template = \"\"\"
Given the following context, answer the question.
Context:
{% for doc in documents %}
  {{ doc.content }}
{% endfor %}
Question: {{ question }}
Answer:
\"\"\"
prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator(model="gpt-4o-mini")

rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# 3. Run Pipeline
question = "What is the capital of France?"
result = rag_pipeline.run({
    "retriever": {"query": question},
    "prompt_builder": {"question": question},
})

# Structure results for rag-eval:
eval_results = [
    {
        "question": question,
        "replies": result["llm"]["replies"],
        "retrieved_documents": result["retriever"]["documents"],
        "golden_documents": ["Paris is the capital and largest city of France."],
    }
]
```
"""

from typing import Any

from rag_eval.models import Sample


def from_haystack(eval_results: list[dict[str, Any]]) -> list[Sample]:
    """Convert Haystack evaluation results to a list of rag-eval Sample objects.

    Args:
        eval_results: List of dictionaries representing Haystack evaluation records.
            Expected fields per dict:
              - 'question' or 'query': The question text (str)
              - 'replies' or 'answer': The LLM generated answer string or list of strings
              - 'retrieved_documents' or 'documents': List of Haystack Document objects or dicts
              - 'golden_documents': Optional list of ground-truth document strings
              - 'id': Optional unique ID string

    Returns:
        List of validated Sample instances ready for evaluation.
    """
    samples: list[Sample] = []
    for item in eval_results:
        question: str = item.get("question") or item.get("query") or ""

        # Extract answer from replies list or direct answer field
        answer_raw = item.get("replies") or item.get("answer") or ""
        if isinstance(answer_raw, list):
            answer = answer_raw[0] if answer_raw else ""
        else:
            answer = str(answer_raw)

        # Extract context and retrieved chunks
        docs_raw = item.get("retrieved_documents") or item.get("documents") or []
        doc_contents: list[str] = []
        for doc in docs_raw:
            if hasattr(doc, "content"):
                doc_contents.append(str(doc.content))
            elif isinstance(doc, dict):
                doc_contents.append(str(doc.get("content", "")))
            elif isinstance(doc, str):
                doc_contents.append(doc)

        # If no documents were retrieved, provide fallback empty representation
        context: list[str] = doc_contents if doc_contents else ["No context provided."]
        retrieved: list[str] | None = doc_contents if doc_contents else None

        golden = item.get("golden_documents")
        golden_docs: list[str] | None = list(golden) if golden is not None else None

        sample_kwargs: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "context": context,
            "golden_documents": golden_docs,
            "retrieved": retrieved,
        }
        if "id" in item and item["id"]:
            sample_kwargs["id"] = str(item["id"])

        samples.append(Sample.model_validate(sample_kwargs))

    return samples
