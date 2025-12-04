import os
import shutil
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data/docs"
CHROMA_PATH = "data/chroma_db"
CHUNK_SIZE = 800  # Document-as-Chunk approach, for capturing context
CHUNK_OVERLAP = 40


def clean_text(text):
    """Cleans up text by replacing non-ASCII characters with their ASCII equivalents"""
    replacements = {
        "\x91": "'",  # opening smart single quote
        "\x92": "'",  # closing smart single quote
        "\x93": '"',  # opening smart double quote
        "\x94": '"',  # closing smart double quote
        "\x96": "-",  # Simple dash
        "\x97": "--",  # Double dash
    }

    for c, r in replacements.items():
        text = text.replace(c, r)

    return text


files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]

# MarkdownHeaderTextSplitter to split by markdown headers
headers_to_split_on = [("#", "1"), ("##", "2")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

# RecursiveCharacterTextSplitter to split by chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

docs = []

for file in files:
    # Docs contain non-ASCII characters, so we need to adapt the encoding
    with open(file, "r", encoding="latin-1") as f:
        # We clean up the text by replacing non-ASCII characters with their ASCII equivalents, important if we want to use small LLMs
        content = clean_text(f.read())

        # Split by markdown headers
        md_docs = md_splitter.split_text(content)

        # Add metadata (file name)
        for doc in md_docs:
            doc.metadata["source"] = file

        # Split by chunks
        chunks = text_splitter.split_documents(md_docs)
        docs.extend(chunks)


print(f"Split into {len(docs)} chunks")

# TODO: Add several models, justify why
# Qwen/Qwen3-Embedding-0.6B is ranked #6 on the HuggingFace MTEB leaderboard, with a memory usage of only 1136MB and 1024 dimensions
# https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
# https://sbert.net/docs/sentence_transformer/pretrained_models.html
# sentence-transformers/all-MiniLM-L6-v2

# 3 - Init embedding model
embedding = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Clean up the Chroma directory if it exists
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# 4 - Create and persist the vector store
# We could use InMemory vectors, but it will be simpler to persist them to disk
vector_store = Chroma.from_documents(
    docs,
    embedding=embedding,
    persist_directory=CHROMA_PATH,
)

print(f"Chroma vector store created at {CHROMA_PATH}")
