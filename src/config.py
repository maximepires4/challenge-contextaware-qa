# Paths
DATA_DIR = "data/docs"
CHROMA_PATH = "data/chroma_db"
QUESTIONS_FILE = "data/questions.json"
RESULTS_FILE = "data/results-final.json"

# Models
MODELS_DIR = "models"

AVAILABLE_CHAT_MODELS = {
    "qwen": {  # Tiny model
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    },
    "gemma": {  # SOTA tiny model
        "repo": "google/gemma-3-1b-it-qat-q4_0-gguf",
        "filename": "gemma-3-1b-it-q4_0.gguf",
    },
    "llama": {  # Llama tiny model
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    },
    "phi": {  # Larger model
        "repo": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
    },
    "exaone": {  # Best IFEval score
        "repo": "lmstudio-community/EXAONE-3.5-2.4B-Instruct-GGUF",
        "filename": "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf",
    },
    "benchmaxx": {  # Best BBH score
        "repo": "mradermacher/Benchmaxx-Llama-3.2-1B-Instruct-GGUF",
        "filename": "Benchmaxx-Llama-3.2-1B-Instruct.Q4_K_M.gguf",
    },
    "granite": {  # Good average score
        "repo": "bartowski/granite-3.1-2b-instruct-GGUF",
        "filename": "granite-3.1-2b-instruct-Q4_K_M.gguf",
    },
}

AVAILABLE_RERANK_MODELS = {
    "bge": {
        "repo": "BAAI/bge-reranker-v2-m3",
        "score_threshold": 0.03,
    },
    "ms-marco": {
        "repo": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "score_threshold": -8,
    },
}

DEFAULT_CHAT_MODEL = "qwen"
DEFAULT_RERANK_MODEL = "bge"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

JUDGE_MODEL_NAME = "gemini-2.5-flash"

# Chunking strategy
CHUNK_SIZE = 800  # Document-as-Chunk approach, for capturing context
CHUNK_OVERLAP = 150

# RAG parameters
MAX_TOKENS = 1024
MAX_TOKENS_SAFE = 1000  # Buffer for safety
TOP_K_RERANK = 3  # Number of documents to keep after reranking
DOC_SEPARATOR = "\n\n---\n\n"

# Prompts
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

JUDGE_TEMPLATE = """
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to evaluate the quality of a generated answer compared to a ground truth answer, taking into account the actual context retrieved by the system.

**Question:** {question}

**Ground Truth Answer:** {ground_truth}
**Expected Sources:** {expected_sources}

**Generated Answer:** {generated_answer}
**Retrieved Sources (Filenames):** {sources}

### RETRIEVED CONTEXT CONTENT
The following text was retrieved by the RAG system. Use this to determine if the answer is grounded in the context (hallucination check) and if the relevant information was found.

{retrieved_context}
"""

SOURCE_CONTENT_FORMAT = """
--- Source: {source} ---
{content}
"""

EVALUATION_SCHEMA = {
    "properties": {
        "correctness": {
            "description": "Does the answer convey the same key information as the ground truth? 5 = Perfect match in meaning. 1 = Completely wrong",
            "anyOf": [
                {"type": "integer", "minimum": 1, "maximum": 5},
                {"type": "null"},
            ],
        },
        "completeness": {
            "description": "Does the answer miss any critical details mentioned in the ground truth? 5 = All details present. 1 = Key details missing.",
            "anyOf": [
                {"type": "integer", "minimum": 1, "maximum": 5},
                {"type": "null"},
            ],
        },
        "recall": {
            "description": "Did the system retrieve the EXPECTED sources? 5 = All expected sources present. 1 = None found.",
            "anyOf": [
                {"type": "integer", "minimum": 1, "maximum": 5},
                {"type": "null"},
            ],
        },
        "precision": {
            "description": 'Is the context "diluted" with too many irrelevant files? 5 = Only relevant files retrieved. 1 = Mostly junk files.',
            "anyOf": [
                {"type": "integer", "minimum": 1, "maximum": 5},
                {"type": "null"},
            ],
        },
        "hallucination": {
            "type": "boolean",
            "description": "Does the answer contain information NOT present in the ground truth or that seems invented?",
        },
        "summary": {
            "type": "string",
            "description": "A brief explanation of the scores",
        },
    },
    "required": [
        "correctness",
        "completeness",
        "recall",
        "precision",
        "hallucination",
        "summary",
    ],
    "title": "Evaluation",
    "type": "object",
}
