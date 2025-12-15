# Architecture

This document explains the architecture of the Context-Aware QA RAG system.

## 1. Overview

The system answers questions in 4 steps:

1. **Retrieve:** Find top-20 relevant text chunks using Vector Search.
2. **Rerank:** Sort these chunks by precision using a Cross-Encoder.
3. **Select:** Keep the best chunks until the 1024 token limit is reached.
4. **Generate:** Produce the answer using a tiny local model (e.g. Qwen 2.5).

---

## 2. Data Ingestion (`src/ingestion.py`)

We treat documents specifically to help the model understand context.

### Cleaning

* **Encoding:** We read files as `latin-1` to avoid crashes on special characters (like Windows smart quotes).
* **Normalization:** We fix bad characters before processing.

### Chunking Strategy

We do not chop text randomly. We use a **Structure-Aware** approach:

1. **Split by Headers:** We split Markdown files by sections (`# Header`).
2. **Context Injection:** We write the filename and headers *inside* each chunk.

### Storage

* **Database:** ChromaDB (Persistent).
* **Embeddings:** `sentence-transformers/all-mpnet-base-v2`.

---

## 3. Retrieval Pipeline (`src/rag.py`)

The context window (1024 tokens) is very small, and the tiny LLM needs precise information not diluted by noise.

### Step A: Vector Search (Recall)

* We get the top **20 documents** from ChromaDB.
* This wide selection is enough to find all potentially relevant documents.

### Step B: Reranking & Filtering

* We use as a **Cross-Encoder** (`BAAI/bge-reranker-v2-m3`), giving a precise score to each documents.
* **Score Thresholding:**
  * We apply a strict cut-off (e.g., Score > 0.2), rejecting everything else.
  * Tiny LLMs are easily tricked by "diluted information". Even slightly irrelevant text can make them hallucinate.
  * By strictly filtering the context, we protect the model from noise.

### Step C: Greedy Selection (Token Management)

We fill the context window carefully:

1. Sort chunks by their Reranker score (best first).
2. Add them to the context if the total tokens of the retrieved chunks are below 1000.

*Note: We use a safe limit of 1000 (not 1024) to keep a safety buffer.*

---

## 4. Generation (`src/main.py`)

* **Engine:** `llama.cpp`.
* **Format:** GGUF (Quantized models for CPU efficiency).
* **Model:** `Qwen2.5-1.5B-Instruct`.
  * Selected for its reasoning capability at a small size.
* **Parameters:** `Temperature = 0` for deterministic answers.
