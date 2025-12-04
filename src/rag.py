import json

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate

MAX_TOKENS = 1024
MAX_TOKENS_SAFE = 1000  # Safety margin, to avoid exceeding the token budget (Token(A) + Token(B) != Token(A + B))
DOC_SEPARATOR = "\n\n---\n\n"  # Separator between chunks of context
CHROMA_PATH = "data/chroma_db"
MODEL_FILE = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# 1. Load resources
print("Loading resources...")

# Emedding model for retrieval
embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Chroma vector store
vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

# LLM for answer generation
llm = ChatLlamaCpp(
    model_path=MODEL_FILE,
    temperature=0,  # 0 for factual and deterministic answers # TODO: 0.1 for avoiding repetition ?
    max_tokens=1024,
    n_ctx=2048,
    verbose=False,
)

# As the context window is limited, we need to keep track of tokens used for separating chunks
doc_separator_tokens = llm.get_num_tokens(DOC_SEPARATOR)

print("Resources loaded.")

# Load questions
with open("data/questions.json", "r") as f:
    questions_data = json.load(f)

questions = questions_data["questions"]

# For a small LLM, we need a strict template to avoid hallucinations
template = """You are a precise technical assistant for the ZentroSoft Space Station.
Your task is to answer the question using ONLY the provided context snippets.

RULES:
1. If the answer is not in the context, say "Insufficient information in context."
2. If the context contains contradictory information, EXPLICITLY mention the contradiction.
3. Do not make up procedures. Stick to the steps listed in the documents.
4. Be concise.

Context:
{context}

Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# 2. Process questions
results = []

for question in questions:
    q_id = question["id"]
    q_text = question["question"]
    print(f"\n\n{'=' * 100}")
    print(f"Processing question {q_id}: {q_text}\n")

    # a. Retrieve context
    docs = vector_store.similarity_search(q_text, k=20)

    # b. Select relevant context
    selected_docs = []
    current_tokens = 0

    for doc in docs:
        content = doc.page_content
        tokens = llm.get_num_tokens(content)

        # Ensure the token budget won't be exceeded
        if current_tokens + tokens < MAX_TOKENS_SAFE:
            selected_docs.append(doc)
            # Add also separator tokens count, as adding a document after will always require a separator
            current_tokens += tokens + doc_separator_tokens
            print(f"    + Selected doc (tokens: {tokens}): {doc.metadata['source']}")
        else:
            # Do not break the loop, as smaller documents might come after
            print(f"    - Skipped doc (tokens: {tokens}): {doc.metadata['source']}")

    # c. Generate answer
    context_text = DOC_SEPARATOR.join([doc.page_content for doc in selected_docs])
    print(f"\nTotal context tokens: {llm.get_num_tokens(context_text)}/{MAX_TOKENS}")

    message = prompt.format(context=context_text, question=q_text)

    response = llm.invoke(message)
    print(f"ANSWER:\n{'-' * 100}\n{response.content}\n{'-' * 100}")

    # d. Save results (answer and context for evaluation)
    results.append(
        {
            "id": q_id,
            "question": q_text,
            "answer": response.content,
            "context": [doc.metadata["source"] for doc in selected_docs],
        }
    )

# 3. Save results
with open("data/results.json", "w") as f:
    json.dump({"answers": results}, f, indent=2)

print("Answers saved to data/results.json")
