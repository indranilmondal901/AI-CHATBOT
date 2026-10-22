"""
Evaluate the Budhana Tech RAG chatbot using RAGAS.

Expected ask_document() output:
{
    "answer": "Generated answer...",
    "context": ["Retrieved chunk 1...", "Retrieved chunk 2..."]
}
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# --------------------------------------------------
# 1. Resolve project paths and load environment
# --------------------------------------------------

# Project root: AI-CHATBOT/
BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

from src.rag import ask_document

dataset_path = Path(__file__).resolve().parent / "test_dataset.json"

with dataset_path.open("r", encoding="utf-8") as file:
    test_cases = json.load(file)

if not test_cases:
    raise ValueError("The evaluation dataset is empty.")

# --------------------------------------------------
# 2. Configure evaluation models
# --------------------------------------------------

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "GOOGLE_API_KEY is missing. Add it to your .env file."
    )

# Gemini evaluates the chatbot's answers.
evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model=os.getenv(
            "RAGAS_EVALUATOR_MODEL",
            "gemini-2.5-flash",
        ),
        temperature=0,
    )
)

# Embeddings are used by RAGAS metrics.
evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)

# --------------------------------------------------
# 3. Normalize chatbot output
# --------------------------------------------------

def get_answer_and_context(result):
    """Extract the generated answer and retrieved text chunks."""

    if not isinstance(result, dict):
        raise TypeError(
            "ask_document() must return a dictionary containing "
            "'answer' and retrieved contexts."
        )

    answer = (
        result.get("answer")
        or result.get("response")
        or result.get("result")
    )

    contexts = (
        result.get("retrieved_contexts")
        or result.get("context")
        or result.get("source_documents")
        or result.get("documents")
        or []
    )

    if isinstance(contexts, str):
        contexts = [contexts]

    normalized_contexts = []

    for item in contexts:
        if isinstance(item, str):
            if item.strip():
                normalized_contexts.append(item)

        elif hasattr(item, "page_content"):
            if item.page_content and item.page_content.strip():
                normalized_contexts.append(item.page_content)

        elif isinstance(item, dict):
            content = item.get("page_content") or item.get("text")
            if content and content.strip():
                normalized_contexts.append(content)

    if hasattr(answer, "content"):
        answer = answer.content

    return answer, normalized_contexts


# --------------------------------------------------
# 4. Run chatbot and collect evaluation records
# --------------------------------------------------

records = []

for index, test in enumerate(test_cases, start=1):
    question = test["question"]
    expected_answer = test.get("expected_answer", "").strip()

    # No document_id: search across the available documents.
    result = ask_document(question=question)

    answer, contexts = get_answer_and_context(result)

    if not answer:
        raise ValueError(
            f"Test case {index}: the RAG chatbot returned no answer."
        )

    if not contexts:
        raise ValueError(
            f"Test case {index}: no retrieved context was returned. "
            "Check ChromaDB, ingestion, and ask_document()."
        )

    if (
        not expected_answer
        or expected_answer.startswith("REPLACE_WITH")
        or expected_answer.startswith("Replace this")
    ):
        raise ValueError(
            f"Test case {index}: replace expected_answer with a "
            "verified reference answer from your approved documents."
        )

    records.append(
        {
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": str(answer),
            "reference": expected_answer,
        }
    )

    print(f"Collected evaluation case {index}/{len(test_cases)}")


# --------------------------------------------------
# 5. Create RAGAS dataset
# --------------------------------------------------

evaluation_dataset = Dataset.from_list(records)

# --------------------------------------------------
# 6. Evaluate RAG quality
# --------------------------------------------------

metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

print("\nRunning RAGAS evaluation...")

results = evaluate(
    dataset=evaluation_dataset,
    metrics=metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

# --------------------------------------------------
# 7. Save and display results
# --------------------------------------------------

scores = results.to_pandas()

output_path = BASE_DIR / "ragas_results.csv"
scores.to_csv(output_path, index=False)

print("\nEvaluation completed.")
print(scores.to_string(index=False))
print(f"\nDetailed scores saved to: {output_path}")