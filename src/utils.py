import os
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
import config


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
    files = [os.path.join(config.DATA_DIR, f) for f in os.listdir(config.DATA_DIR)]

    # MarkdownHeaderTextSplitter to split by markdown headers
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    # RecursiveCharacterTextSplitter to split by chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    docs = []

    for file in files:
        # Docs contain non-ASCII characters, so we need to adapt the encoding
        with open(file, "r", encoding="latin-1") as f:
            # We clean up the text by replacing non-ASCII characters with their ASCII equivalents, important if we want to use small LLMs
            content = clean_text(f.read())

            # Split by markdown headers
            md_docs = md_splitter.split_text(content)

            # Add metadata (file name) and inject headers into content
            for doc in md_docs:
                doc.metadata["source"] = file

                # Re-inject headers into the content so embeddings/LLM see the context
                header_context = ""
                if "Header 1" in doc.metadata:
                    header_context += f"# {doc.metadata['Header 1']}\n"
                if "Header 2" in doc.metadata:
                    header_context += f"## {doc.metadata['Header 2']}\n"

                if header_context:
                    doc.page_content = f"{header_context}\n{doc.page_content}"

            # Split by chunks
            chunks = text_splitter.split_documents(md_docs)
            docs.extend(chunks)

    return docs

