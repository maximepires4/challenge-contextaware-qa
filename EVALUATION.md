# Evaluation Report

This document analyzes the test results.

## Methodology

We compared answers from local models against answers from a superior model (Gemini 3.0 Pro).

**Criteria:**

1. **Correctness:** Does the answer convey the same meaning as the ground truth? (1-5)
2. **Completeness:** Are all key details present? (1-5)
3. **Recall:** Did the system find all expected source documents? (1-5)
4. **Precision:** Is the context free of irrelevant files? (1-5)
5. **Hallucination:** Does the answer contain invented information? (True/False)

---

## 1. Retrieval Strategy Analysis

We tested different ways to find documents using the Qwen model.

### Vector Search

* **Config:** Top-15 documents, no reranking.
* **Observation:** Too many documents (10+). Relevant info is diluted.
* **Analysis:** The model fails to find fine contradictions (Q3) and invents procedures (Q4) because there is too much noise.

### Hybrid Search

* **Config:** Keyword Search (BM25) + Vector Search.
* **Analysis:** Keywords do not help here because the questions are about concepts, not specific terms. Results are the same as Vector Search.

### Reranking

* **Config:** Top-20 Vector Search -> Cross-Encoder -> Top-3 Selection.
* **Analysis:** Best precision. Only essential paragraphs are kept. The model correctly handles ambiguity (Q2) and logic (Q3).

---

## 2. Model Benchmark

Using the Reranking strategy, we tested 7 local models.

### Qwen2.5-1.5B-Instruct

* **Why:** Industry standard for tiny models.
* **Performance:** Best reasoning. Good Markdown formatting.
* **Weakness:** Invents steps for missing procedures (Q4).

### Gemma-2-2b-it

* **Why:** Chosen for its "Good average scores" on standard benchmarks.
* **Performance:** Robust. It does not hallucinate on Q4 (it just repeats the instruction).
* **Weakness:** Less detailed than Qwen on complex procedures (Q1).

### Phi-3-mini-4k

* **Why:** Included as a "Larger model" to see if size matters.
* **Performance:** Very rigorous. No invented facts.
* **Weakness:** Too rigid. Refuses to answer Q2 ("Insufficient information") even when the answer is in the text.

### Llama-3.2-1B-Instruct

* **Why:** Another industry standard (but from LLama).
* **Performance:** Fluid answers but too long.
* **Weakness:** Wastes tokens by repeating the question. Logical error on Q3 (misunderstood "Isolation").

### EXAONE-3.5-2.4B

* **Why:** Selected for its "Best IFEval score" (instruction following).
* **Performance:** Structured answers.
* **Weakness:** Bad hallucinations. Invents fake system commands (`reboot -p now`, `/opt/zshell/...`).

### Granite-3.1-2b

* **Why:** Another candidate with "Good average score".
* **Performance:** Weak.
* **Weakness:** Repeats itself in loops. Complicated sentences with no info.

### Llama-3.2-Benchmaxx

* **Why:** Included for its "Best BBH score" (complex reasoning).
* **Performance:** Incoherent answers.

