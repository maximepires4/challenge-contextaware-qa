import json
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import config
import utils

# 1. Load resources
print("Loading resources...")

# A. Build BM25 Index (Hybrid Search Component)
print("Building BM25 index...")
# We use the shared utility to ensure BM25 sees exactly the same chunks as Chroma
bm25_docs = utils.load_and_split_docs()
tokenized_corpus = [doc.page_content.split() for doc in bm25_docs]
bm25 = BM25Okapi(tokenized_corpus)

# B. Standard RAG components
embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

# Chroma vector store
vector_store = Chroma(
    persist_directory=config.CHROMA_PATH, embedding_function=embedding_model
)

# LLM for answer generation
llm = ChatLlamaCpp(
    model_path=config.MODEL_FILE,
    temperature=0,  # 0 for factual and deterministic answers
    max_tokens=config.MAX_TOKENS,
    n_ctx=2048,
    verbose=False,
)

# Reranker for context selection
reranker = CrossEncoder(config.RERANK_MODEL_NAME)

# As the context window is limited, we need to keep track of tokens used for separating chunks
doc_separator_tokens = llm.get_num_tokens(config.DOC_SEPARATOR)

print("Resources loaded.")

# Load questions
with open(config.QUESTIONS_FILE, "r") as f:
    questions_data = json.load(f)

questions = questions_data["questions"]

# For a small LLM, we need a strict template to avoid hallucinations
prompt = ChatPromptTemplate.from_template(config.STRICT_TEMPLATE)

# 2. Process questions
results = []

for question in questions:
    q_id = question["id"]
    q_text = question["question"]
    print(f"\n\n{'=' * 100}")
    print(f"Processing question {q_id}: {q_text}\n")

    # A. Hybrid retrieval

    # BM25 retrieval (Top 1 VIP)
    tokenized_query = q_text.split()
    bm25_top_docs = bm25.get_top_n(tokenized_query, bm25_docs, n=1)
    vip_doc = bm25_top_docs[0] if bm25_top_docs else None

    # Vector retrieval (Top 20)
    vector_docs = vector_store.similarity_search(q_text, k=20)

    # B. Reranking (Vector results only)
    pairs = [[q_text, doc.page_content] for doc in vector_docs]
    scores = reranker.predict(pairs)

    # Combine docs with their scores and sort by score descending
    docs_with_scores = list(zip(vector_docs, scores))
    docs_with_scores.sort(key=lambda x: x[1], reverse=True)

    # C. Context selection (BM25 VIP + Best reranked)
    selected_docs = []
    current_tokens = 0
    included_contents = set()

    # Force include BM25 VIP Doc
    if vip_doc:
        selected_docs.append(vip_doc)
        # Add also separator tokens count
        tokens = llm.get_num_tokens(vip_doc.page_content)
        current_tokens += tokens + doc_separator_tokens
        included_contents.add(vip_doc.page_content)
        print(
            f"    + Selected (BM25 VIP) | Tokens: {tokens} | {vip_doc.metadata['source']}"
        )

    # 2. Fill remaining budget with vector docs
    for doc, score in docs_with_scores:
        content = doc.page_content

        # Avoid duplicates (if BM25 found the same doc as vector)
        if content in included_contents:
            continue

        tokens = llm.get_num_tokens(content)

        # Ensure the token budget won't be exceeded
        if current_tokens + tokens > config.MAX_TOKENS_SAFE:
            print(
                f"    - Skipped (budget) | Score: {score:.4f} Tokens: {tokens} | {doc.metadata['source']}"
            )
            # Do not break the loop, as smaller documents might come after
            continue

        if score < config.SCORE_THRESHOLD:
            print(
                f"    - Skipped (low score) | Score: {score:.4f} Tokens: {tokens} | {doc.metadata['source']}"
            )
            # We could break the loop as the list is sorted, but for output clarity we keep it
            continue

        selected_docs.append(doc)
        # Add also separator tokens count, as adding a document after will always require a separator
        current_tokens += tokens + doc_separator_tokens
        included_contents.add(content)
        print(
            f"    + Selected (Vector) | Score: {score:.4f} Tokens: {tokens} | {doc.metadata['source']}"
        )

    # d. Generate answer
    context_text = config.DOC_SEPARATOR.join(
        [doc.page_content for doc in selected_docs]
    )
    print(
        f"\nTotal context tokens: {llm.get_num_tokens(context_text)}/{config.MAX_TOKENS}"
    )

    message = prompt.format(context=context_text, question=q_text)

    response = llm.invoke(message)
    print(f"ANSWER:\n{'-' * 100}\n{response.content}\n{'-' * 100}")

    # e. Save results (answer and context for evaluation)
    results.append(
        {
            "id": q_id,
            "question": q_text,
            "answer": response.content,
            "context": [doc.metadata["source"] for doc in selected_docs],
        }
    )

# 3. Save results
with open(config.RESULTS_FILE, "w") as f:
    json.dump({"answers": results}, f, indent=2)

print(f"Answers saved to {config.RESULTS_FILE}")