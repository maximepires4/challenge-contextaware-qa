import os
import argparse
import time

from google import genai
from dotenv import load_dotenv

import config
from utils import load_json


def evaluate_answer(
    question, ground_truth, generated_answer, expected_sources, sources, source_contents
):
    """
    Uses Gemini to evaluate the generated answer against the ground truth.
    """

    # Build context string from retrieved sources
    retrieved_context_str = ""
    for src in sources:
        retrieved_context_str += config.SOURCE_CONTENT_FORMAT.format(
            source=src, content=source_contents[src]
        )

    # Build prompt
    # New Google SDK separates the prompt from the response schema, indicating specificly that no schema information should be included in th prompt
    prompt = config.JUDGE_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        expected_sources=", ".join(expected_sources),
        generated_answer=generated_answer,
        sources=", ".join(sources),
        retrieved_context=retrieved_context_str,
    )

    # Initialize Gemini client
    # The API key is loaded from the environment variable GEMINI_API_KEY
    client = genai.Client()

    try:
        # Call Gemini API
        response = client.models.generate_content(
            model=config.JUDGE_MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": config.EVALUATION_SCHEMA,
            },
        )
        return response.parsed
    except Exception as e:
        # Fallback (for example, if API quotas are exceeded)
        print(f"Error calling Gemini API: {e}")
        return {
            "correctness": 0,
            "completeness": 0,
            "recall": 0,
            "precision": 0,
            "hallucination": True,
            "summary": f"API Error: {e}",
        }
    finally:
        # Close Gemini client
        # We could implement a threaded mechanism on the main loop, but this is simpler
        client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG results using Gemini as a Judge."
    )
    parser.add_argument(
        "--ground-truth",
        default="data/ground_truth.json",
        help="Path to ground truth JSON",
    )
    parser.add_argument(
        "--results", default="data/results.json", help="Path to generated results JSON"
    )

    args = parser.parse_args()

    # Load environment variables from .env file, which should include the Gemini API key
    load_dotenv()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not found.")
        return

    print("Loading data...")
    ground_truth_data = load_json(args.ground_truth)
    results_data = load_json(args.results)

    # Load all source documents into memory to inject into context for evaluation
    print("Loading source documents...")
    source_contents = {}
    if os.path.exists(config.DOCS_DIR):
        for filename in os.listdir(config.DOCS_DIR):
            if filename.endswith(".md"):
                path = os.path.join(config.DOCS_DIR, filename)
                with open(path, "r", encoding="latin-1") as f:
                    source_contents[path] = f.read()

    print(f'\nEvaluating answers using "{config.JUDGE_MODEL_NAME}":\n')

    for res in results_data["answers"]:
        q_id = res["id"]

        # Find corresponding ground truth question
        gt = None
        for item in ground_truth_data["questions"]:
            if item["id"] == q_id:
                gt = item
                break

        print(f"Evaluating question {q_id}...")

        eval_result = evaluate_answer(
            question=gt["question"],
            ground_truth=gt["answer"],
            generated_answer=res["answer"],
            expected_sources=gt["sources"],
            sources=res["context"],
            source_contents=source_contents,
        )

        # Console output
        print(
            f"Scores: Correctness={eval_result['correctness']}/5, Completeness={eval_result['completeness']}/5"
        )
        print(
            f"Sources: Recall={eval_result['recall']}/5, Precision={eval_result['precision']}/5"
        )
        if eval_result["hallucination"]:
            print("Hallucination detected")
        else:
            print("No hallucination detected")
        print(f"Summary: {eval_result['summary']}")
        print(f"\n{'-' * 40}\n")

        time.sleep(1)  # Avoid API quotas


if __name__ == "__main__":
    main()
