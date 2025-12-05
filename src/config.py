# Paths
DATA_DIR = "data/docs"
CHROMA_PATH = "data/chroma_db"
QUESTIONS_FILE = "data/questions.json"
RESULTS_FILE = "data/results-final.json"

# Models
MODEL_FILE = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunking strategy
CHUNK_SIZE = 800  # Document-as-Chunk approach, for capturing context
CHUNK_OVERLAP = 150

# RAG parameters
MAX_TOKENS = 1024
MAX_TOKENS_SAFE = 1000  # Buffer for safety
SCORE_THRESHOLD = -8  # Strict threshold for vector results
DOC_SEPARATOR = "\n\n---\n\n"

# Prompt
STRICT_TEMPLATE = """### INSTRUCTION
You are a strict technical assistant for ZentroSoft. 
You answer questions based SOLELY on the context below.

### STRICT RULES
1. NO OUTSIDE KNOWLEDGE: Never use your own training data. If the answer is not in the context, say "Insufficient information".
2. QUOTE PROCEDURES: If asked for a procedure, list the steps EXACTLY as written in the text. Do not paraphrase or invent steps.
3. CONTRADICTIONS: If documents disagree, mention both versions explicitly.

### CONTEXT
{context}

### USER QUESTION
{question}

### ANSWER
"""
