import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import config
import utils


def run_ingestion(embedding_model_name=config.EMBEDDING_MODEL_NAME, verbose=False):
    """
    Main ingestion function.
    Loads docs, splits them, and creates the Chroma vector store.
    """
    # 1 - Load and split documents
    # Logic moved to utils.py to be shared with rag.py (BM25 needs the same chunks)
    if verbose:
        print(f"Loading documents from {config.DATA_DIR}...")
    docs = utils.load_and_split_docs()

    if verbose:
        print(f"Split into {len(docs)} chunks")

    # 3 - Init embedding model
    if verbose:
        print(f"Loading embedding model: {embedding_model_name}...")
    embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Clean up the Chroma directory if it exists
    if os.path.exists(config.CHROMA_PATH):
        if verbose:
            print(f"Removing existing DB at {config.CHROMA_PATH}...")
        shutil.rmtree(config.CHROMA_PATH)

    # 4 - Create and persist the vector store
    # We could use InMemory vectors, but it will be simpler to persist them to disk
    if verbose:
        print(f"Creating Chroma vector store at {config.CHROMA_PATH}...")
    Chroma.from_documents(
        docs,
        embedding=embedding,
        persist_directory=config.CHROMA_PATH,
    )

    if verbose:
        print(f"Ingestion complete. DB saved to {config.CHROMA_PATH}")


if __name__ == "__main__":
    run_ingestion()
