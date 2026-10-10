import json
from pathlib import Path

from dotenv import load_dotenv
from langsmith import evaluate

load_dotenv()

# Import the RAG chain without starting the interactive loop
from test.rag import rag_chain


# 1. Load our local test dataset
dataset_path = (
    Path(__file__).resolve().parent / "test_dataset.json"
)

with open(dataset_path, "r", encoding="utf-8") as file:
    test_cases = json.load(file)


# 2. Convert the test cases to LangSmith examples
examples = [
    {
        "inputs": {
            "question": test["question"]
        },
        "outputs": {
            "expected_answer": test["expected_answer"],
            "expected_keywords": test["expected_keywords"],
            "should_refuse": test["should_refuse"]
        }
    }
    for test in test_cases
]


# 3. Define the chatbot's prediction function
def predict(inputs: dict) -> dict:
    question = inputs["question"]

    response = rag_chain.invoke(question)

    return {
        "answer": response.content
    }


# 4. Define a simple rule-based evaluator
def basic_quality(outputs: dict, reference_outputs: dict) -> dict:
    answer = outputs.get("answer", "").lower()

    expected_keywords = reference_outputs.get(
        "expected_keywords", []
    )
    should_refuse = reference_outputs.get(
        "should_refuse", False
    )

    if should_refuse:
        refusal_phrases = [
            "not available",
            "not found",
            "don't have",
            "do not have",
            "cannot find",
            "does not contain"
        ]

        passed = any(
            phrase in answer for phrase in refusal_phrases
        )

        return {
            "key": "basic_quality",
            "score": 1.0 if passed else 0.0,
            "comment": (
                "Missing-information response detected."
                if passed
                else "Expected the chatbot to acknowledge missing information."
            )
        }

    if not expected_keywords:
        return {
            "key": "basic_quality",
            "score": 0.0,
            "comment": "No expected keywords were provided."
        }

    matched = [
        keyword for keyword in expected_keywords
        if keyword.lower() in answer
    ]

    score = len(matched) / len(expected_keywords)

    return {
        "key": "basic_quality",
        "score": score,
        "comment": (
            f"Matched {len(matched)} of "
            f"{len(expected_keywords)} expected keywords."
        )
    }


# 5. Run the evaluation experiment
results = evaluate(
    predict,
    data=examples,
    evaluators=[basic_quality],
    experiment_prefix="budhana-rag-baseline",
    description="Baseline evaluation of the Budhana Tech RAG chatbot"
)

print("\nEvaluation completed.")
print("Review the experiment and individual scores in LangSmith.")