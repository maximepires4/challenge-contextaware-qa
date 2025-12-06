import argparse
import config
import utils
from rag import RAGPipeline
from termcolor import colored, cprint


def main():
    parser = argparse.ArgumentParser(description="Context-Aware QA System CLI")

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["chat", "batch", "rerank"],
        default="chat",
        help="Run mode: 'chat' for interactive session, 'batch' for processing questions.json, 'rerank' for evaluating reranker",
    )

    # Configuration
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_CHAT_MODEL,
        choices=list(config.AVAILABLE_CHAT_MODELS.keys()),
        help=f"Chat model to use. Available: {', '.join(config.AVAILABLE_CHAT_MODELS.keys())}",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default=config.DEFAULT_RERANK_MODEL,
        choices=list(config.AVAILABLE_RERANK_MODELS.keys()),
        help=f"Reranker model to use. Available: {', '.join(config.AVAILABLE_RERANK_MODELS.keys())}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON file for batch results",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=None,
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Chat model: Download if needed and get path
    model_path = utils.ensure_model_exists(args.model)

    cprint("Initializing RAG Pipeline...", "cyan", attrs=["bold"])
    print(f"  - Chat Model: {colored(args.model, 'yellow')} ({model_path})")
    print(f"  - Reranker: {colored(args.reranker, 'yellow')}")
    print(f"  - Embedding: {colored(config.EMBEDDING_MODEL_NAME, 'yellow')}")

    verbose = args.verbose if args.verbose is not None else args.mode != "chat"

    # Initialize pipeline
    rag = RAGPipeline(
        model_path=model_path,
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        rerank_config=config.AVAILABLE_RERANK_MODELS[args.reranker],
        verbose=verbose,
    )

    if args.mode != "chat":
        if len(args.output) == 0:
            output = f"{config.DATA_DIR}/results-{args.model}.json"
        else:
            output = args.output

        if args.mode == "batch":
            cprint(
                f"Running in Batch Mode (output: {output})",
                "magenta",
                attrs=["bold"],
            )
        elif args.mode == "rerank":
            cprint(
                "Running in Reranker Evaluation Mode (no output)",
                "magenta",
                attrs=["bold"],
            )

        rag.run_batch(config.QUESTIONS_FILE, output, answer=args.mode == "batch")

    else:
        print("\n" + colored("=" * 50, "green"))
        cprint("ZentroSoft Technical Assistant", "green", attrs=["bold"])
        cprint("Type 'exit' or 'quit' to stop.", "dark_grey")
        print(colored("=" * 50, "green") + "\n")

        while True:
            try:
                user_input = input(colored("You: ", "blue", attrs=["bold"]))
                if user_input.lower() in ["exit", "quit"]:
                    cprint("Goodbye!", "cyan")
                    break

                if not user_input.strip():
                    continue

                rag.answer_question(user_input)

            except KeyboardInterrupt:
                cprint("\nGoodbye!", "cyan")
                break
            except Exception as e:
                cprint(f"\nError: {e}", "red")


if __name__ == "__main__":
    main()
