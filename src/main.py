import argparse
import config
from rag import RAGPipeline
from termcolor import colored, cprint


def main():
    parser = argparse.ArgumentParser(description="Context-Aware QA System CLI")

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["chat", "batch"],
        default="chat",
        help="Run mode: 'chat' for interactive session, 'batch' for processing questions.json",
    )

    # Configuration
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_FILE,
        help="Path to the GGUF model file",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=config.EMBEDDING_MODEL_NAME,
        help="HuggingFace embedding model name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=config.RESULTS_FILE,
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

    cprint("Initializing RAG Pipeline...", "cyan", attrs=["bold"])
    print(f"  - Model: {colored(args.model, 'yellow')}")
    print(f"  - Embedding: {colored(args.embedding, 'yellow')}")

    verbose = args.verbose if args.verbose is not None else args.mode == "batch"

    # Initialize pipeline
    rag = RAGPipeline(
        model_path=args.model, embedding_model_name=args.embedding, verbose=verbose
    )

    if args.mode == "batch":
        cprint(
            f"Running in Batch Mode (output: {args.output})", "magenta", attrs=["bold"]
        )
        rag.run_batch(config.QUESTIONS_FILE, args.output)

    elif args.mode == "chat":
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

