import os
import json
from huggingface_hub import hf_hub_download
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
import config


def ensure_model_exists(model_key):
    """
    Checks if the model exists locally, downloads it if not.
    """
    model_info = config.AVAILABLE_CHAT_MODELS[model_key]
    repo = model_info["repo"]
    filename = model_info["filename"]
    path = os.path.join(config.MODELS_DIR, filename)

    if not os.path.exists(config.MODELS_DIR):
        os.makedirs(config.MODELS_DIR)

    print(path, os.path.exists(path))

    if not os.path.exists(path):
        print(f"Model {model_key} not found at {path}.")
        print(f"Downloading {filename} from {repo}...")
        try:
            cached_path = hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=config.MODELS_DIR,
                local_dir_use_symlinks=False,
            )
            print(f"Model downloaded to {cached_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    return path


def clean_text(text):
    """Cleans up text by replacing non-ASCII characters with their ASCII equivalents"""
    replacements = {
        "\x91": "'",  # opening smart single quote
        "\x92": "'",  # closing smart single quote
        "\x93": '"',  # opening smart double quote
        "\x94": '"',  # closing smart double quote
        "\x96": "-",  # simple dash
        "\x97": "--",  # double dash
    }

    for c, r in replacements.items():
        text = text.replace(c, r)

    return text


def load_and_split_docs():
    """
    Loads, cleans, and splits documents.
    Refactored from ingestion.py to be shared with rag.py (for BM25).
    """
    files = [os.path.join(config.DOCS_DIR, f) for f in os.listdir(config.DOCS_DIR)]

    # MarkdownHeaderTextSplitter to split by markdown headers
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    # RecursiveCharacterTextSplitter to split by chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    docs = []

    for file in files:
        filename = os.path.basename(file)

        # Docs contain non-ASCII characters, so we need to adapt the encoding
        with open(file, "r", encoding="latin-1") as f:
            # We clean up the text by replacing non-ASCII characters with their ASCII equivalents, important if we want to use small LLMs
            content = clean_text(f.read())

            # Split by markdown headers
            md_docs = md_splitter.split_text(content)

            # Add metadata (file name) and inject headers + filename into content
            for doc in md_docs:
                doc.metadata["source"] = file

                # Re-inject headers into the content so embeddings/LLM see the context
                header_context = f"Source Document: {filename}\n"

                if "Header 1" in doc.metadata:
                    header_context += f"# {doc.metadata['Header 1']}\n"
                if "Header 2" in doc.metadata:
                    header_context += f"## {doc.metadata['Header 2']}\n"

                doc.page_content = f"{header_context}\n{doc.page_content}"

            # Split by chunks
            chunks = text_splitter.split_documents(md_docs)
            docs.extend(chunks)

    return docs


def load_json(file_path):
    """Loads a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
