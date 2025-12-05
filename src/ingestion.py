import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import config
import utils


# 1 - Load and split documents
# Logic moved to utils.py to be shared with rag.py (BM25 needs the same chunks)
docs = utils.load_and_split_docs()

print(f"Split into {len(docs)} chunks")

# TODO: Add several models, justify why
# Qwen/Qwen3-Embedding-0.6B is ranked #6 on the HuggingFace MTEB leaderboard, with a memory usage of only 1136MB and 1024 dimensions
# https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
# https://sbert.net/docs/sentence_transformer/pretrained_models.html
# sentence-transformers/all-MiniLM-L6-v2

# 3 - Init embedding model
embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

# Clean up the Chroma directory if it exists
if os.path.exists(config.CHROMA_PATH):
    shutil.rmtree(config.CHROMA_PATH)

# 4 - Create and persist the vector store
# We could use InMemory vectors, but it will be simpler to persist them to disk
vector_store = Chroma.from_documents(
    docs,
    embedding=embedding,
    persist_directory=config.CHROMA_PATH,
)

print(f"Chroma vector store created at {config.CHROMA_PATH}")