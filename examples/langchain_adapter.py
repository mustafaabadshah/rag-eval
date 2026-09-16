"""LangChain evaluation dataset adapter for rag-eval.

Example LangChain LCEL RAG pipeline producing evaluation records:

```python
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. Setup Retriever
vectorstore = FAISS.from_texts(
    ["Paris is the capital of France."],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

# 2. Build RAG Chain
template = \"\"\"Answer the question based only on the following context:
{context}

Question: {question}
\"\"\"
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4o-mini")

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 3. Invoke Chain and Record Outputs
query = "What is the capital of France?"
retrieved_docs = retriever.invoke(query)
answer = rag_chain.invoke(query)

dataset = [
    {
        "query": query,
        "result": answer,
        "source_documents": retrieved_docs,
        "ground_truth": ["Paris is the capital of France."],
    }
]
```
"""

from typing import Any

from rag_eval.models import Sample


def from_langchain(dataset: list[dict[str, Any]]) -> list[Sample]:
    """Convert LangChain evaluation records to a list of rag-eval Sample objects.

    Args:
        dataset: List of dictionaries representing LangChain RAG pipeline outputs.
            Expected keys per dict:
              - 'question' or 'query': User question (str)
              - 'answer' or 'result': Generated answer (str)
              - 'source_documents' or 'context': List of Document objects or strings
              - 'ground_truth' or 'ground_truths': Optional ground-truth document strings
              - 'id': Optional unique ID string

    Returns:
        List of validated Sample instances.
    """
    samples: list[Sample] = []
    for item in dataset:
        question: str = item.get("question") or item.get("query") or ""
        answer: str = str(item.get("answer") or item.get("result") or "")

        docs_raw = item.get("source_documents") or item.get("context") or []
        doc_contents: list[str] = []
        for doc in docs_raw:
            if hasattr(doc, "page_content"):
                doc_contents.append(str(doc.page_content))
            elif isinstance(doc, dict):
                doc_contents.append(str(doc.get("page_content") or doc.get("content") or ""))
            elif isinstance(doc, str):
                doc_contents.append(doc)

        context: list[str] = doc_contents if doc_contents else ["No context provided."]
        retrieved: list[str] | None = doc_contents if doc_contents else None

        golden_raw = item.get("ground_truth") or item.get("ground_truths") or item.get("golden_documents")
        golden_docs: list[str] | None = None
        if golden_raw is not None:
            if isinstance(golden_raw, list):
                golden_docs = [str(g) for g in golden_raw]
            else:
                golden_docs = [str(golden_raw)]

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
