
"""
Evaluate the Budhana Tech RAG chatbot using RAGAS.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
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
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper

# --------------------------------------------------
# 1. Paths and environment
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

from src.rag import ask_document

DATASET_PATH = EVALUATION_DIR / "test_dataset.json"

# IMPORTANT: Use this same filename in your FastAPI endpoint.
OUTPUT_PATH = BASE_DIR / "ragas_results.csv"

with DATASET_PATH.open("r", encoding="utf-8") as file:
    test_cases = json.load(file)

if not test_cases:
    raise ValueError("The evaluation dataset is empty.")

if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "GOOGLE_API_KEY is missing from your .env file."
    )

# --------------------------------------------------
# 2. Configure evaluator
# --------------------------------------------------

evaluator_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model=os.getenv(
            "RAGAS_EVALUATOR_MODEL",
            "gemini-3.8-flash",
        ),
        temperature=0,
    )
)

# evaluator_llm = LangchainLLMWrapper(
#     ChatOllama(
#         model="deepseek-r1:7b",
#         temperature=0,
#     )
# )

evaluator_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
)

# --------------------------------------------------
# 3. Normalize answer and context
# --------------------------------------------------

def normalize_text(value):
    """Convert strings, message objects, and text lists to plain text."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        if value.get("text"):
            return normalize_text(value["text"])

        if value.get("content"):
            return normalize_text(value["content"])

        if value.get("page_content"):
            return normalize_text(value["page_content"])

        return ""

    if isinstance(value, (list, tuple)):
        parts = [normalize_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()

    if hasattr(value, "content"):
        return normalize_text(value.content)

    if hasattr(value, "page_content"):
        return normalize_text(value.page_content)

    return str(value).strip()


def get_answer_and_context(result):
    if not isinstance(result, dict):
        raise TypeError(
            "ask_document() must return a dictionary."
        )

    answer = normalize_text(
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

    if not isinstance(contexts, (list, tuple)):
        contexts = [contexts]

    normalized_contexts = [
        normalize_text(item)
        for item in contexts
    ]

    normalized_contexts = [
        item for item in normalized_contexts if item
    ]

    return answer, normalized_contexts


# --------------------------------------------------
# 4. Collect chatbot responses
# --------------------------------------------------

records = []

for index, test in enumerate(test_cases, start=1):
    question = test.get("question", "").strip()
    expected_answer = normalize_text(
        test.get("expected_answer")
    )

    if not question:
        raise ValueError(
            f"Test case {index} has no question."
        )

    if (
        not expected_answer
        or expected_answer.startswith("REPLACE_WITH")
        or expected_answer.startswith("Replace this")
    ):
        raise ValueError(
            f"Test case {index} needs a verified expected_answer."
        )

    # Search the currently indexed knowledge base.
    result = ask_document(question=question)

    answer, contexts = get_answer_and_context(result)

    if not answer:
        raise ValueError(
            f"Test case {index}: chatbot returned no answer."
        )

    if not contexts:
        raise ValueError(
            f"Test case {index}: no retrieved contexts returned."
        )

    records.append({
        "user_input": question,
        "retrieved_contexts": contexts,
        "response": answer,
        "reference": expected_answer,
    })

    print(
        f"Collected case {index}/{len(test_cases)}"
    )

# --------------------------------------------------
# 5. Create evaluation dataset
# --------------------------------------------------

evaluation_dataset = Dataset.from_list(records)

# --------------------------------------------------
# 6. Run RAGAS evaluation
# --------------------------------------------------

ragas_metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

print("\nRunning RAGAS evaluation...")
print("Evaluation errors will be raised instead of silently ignored.")

results = evaluate(
    dataset=evaluation_dataset,
    metrics=ragas_metrics,
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    raise_exceptions=True,
)

# --------------------------------------------------
# 7. Validate and save results
# --------------------------------------------------

scores = results.to_pandas()

metric_columns = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

missing_columns = [
    column for column in metric_columns
    if column not in scores.columns
]

if missing_columns:
    raise RuntimeError(
        f"RAGAS did not return expected metric columns: "
        f"{missing_columns}. Returned columns: "
        f"{scores.columns.tolist()}"
    )

# Ensure metric values are numeric.
for column in metric_columns:
    scores[column] = pd.to_numeric(
        scores[column], errors="coerce"
    )

# Save the detailed evaluation results.
scores.to_csv(OUTPUT_PATH, index=False)

print("\nEvaluation results:")
print(scores.to_string(index=False))

print(f"\nSaved results to: {OUTPUT_PATH}")

print("\nMetric summary:")
print(scores[metric_columns].mean().to_string())

if scores[metric_columns].notna().sum().sum() == 0:
    raise RuntimeError(
        "RAGAS returned no numeric metric scores. "
        "Check the evaluation error above, your RAGAS version, "
        "Gemini configuration, and evaluator logs."
    )

print("\nRAGAS evaluation completed.")
