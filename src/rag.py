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
import ingestion


class RAGPipeline:
    def __init__(
        self,
        model_path=config.MODEL_FILE,
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        verbose=False,
    ):
        """
        Initializes the RAG pipeline resources.
        Checks for DB existence and runs ingestion if missing.
        """
        self.embedding_model_name = embedding_model_name
        self.verbose = verbose

        # Check for Chroma DB existence
        if not os.path.exists(config.CHROMA_PATH) or not os.listdir(config.CHROMA_PATH):
            if self.verbose:
                print(
                    f"Chroma DB not found at {config.CHROMA_PATH}. Running ingestion..."
                )
            ingestion.run_ingestion(embedding_model_name=self.embedding_model_name)

        if self.verbose:
            print("Loading resources...")

        # A. Build BM25 index (Hybrid Search Component)
        if self.verbose:
            print("Building BM25 index...")
        # We use the shared utility to ensure BM25 sees exactly the same chunks as Chroma
        self.bm25_docs = utils.load_and_split_docs()
        tokenized_corpus = [doc.page_content.split() for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # B. Standard RAG Components
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )

        # Chroma vector store
        self.vector_store = Chroma(
            persist_directory=config.CHROMA_PATH,
            embedding_function=self.embedding_model,
        )

        # LLM for answer generation
        if self.verbose:
            print(f"Loading LLM from {model_path}...")
        self.llm = ChatLlamaCpp(
            model_path=model_path,
            temperature=0,  # 0 for factual and deterministic answers
            max_tokens=config.MAX_TOKENS,
            n_ctx=2048,
            verbose=False,
        )

        # Reranker for context selection
        self.reranker = CrossEncoder(config.RERANK_MODEL_NAME)

        # As the context window is limited, we need to keep track of tokens used for separating chunks
        self.doc_separator_tokens = self.llm.get_num_tokens(config.DOC_SEPARATOR)

        # Strict prompt to avoid hallucinations
        self.prompt = ChatPromptTemplate.from_template(config.STRICT_TEMPLATE)

        if self.verbose:
            print("Resources loaded.")

    def retrieve_context(self, query):
        """
        Performs Hybrid retrieval (BM25 VIP + Vector + Reranking).
        Returns the list of selected Document objects.
        """
        # A. Hybrid retrieval

        # BM25 retrieval (Top 1 VIP)
        tokenized_query = query.split()
        bm25_top_docs = self.bm25.get_top_n(tokenized_query, self.bm25_docs, n=1)
        vip_doc = bm25_top_docs[0] if bm25_top_docs else None

        # Vector retrieval (Top 20)
        vector_docs = self.vector_store.similarity_search(query, k=20)

        # B. Reranking (Vector results only)
        pairs = [[query, doc.page_content] for doc in vector_docs]
        scores = self.reranker.predict(pairs)

        # Combine docs with their scores and sort by score descending
        docs_with_scores = list(zip(vector_docs, scores))
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        # C. Context selection (BM25 VIP + Best reranked)
        selected_docs = []
        current_tokens = 0
        included_contents = set()

        # 1. Force include BM25 VIP Doc
        if vip_doc:
            selected_docs.append(vip_doc)
            tokens = self.llm.get_num_tokens(vip_doc.page_content)
            current_tokens += tokens + self.doc_separator_tokens
            included_contents.add(vip_doc.page_content)
            if self.verbose:
                print(
                    f"    + Selected (BM25 VIP) | Tokens: {tokens} | {vip_doc.metadata['source']}"
                )

        # 2. Fill remaining budget with Vector Docs
        for doc, score in docs_with_scores:
            content = doc.page_content

            # Avoid duplicates (if BM25 found the same doc as Vector)
            if content in included_contents:
                continue

            tokens = self.llm.get_num_tokens(content)

            # Ensure the token budget won't be exceeded
            if current_tokens + tokens > config.MAX_TOKENS_SAFE:
                if self.verbose:
                    print(
                        f"    - Skipped (budget) | Score: {score:.4f} Tokens: {tokens} | {doc.metadata['source']}"
                    )
                # Do not break the loop, as smaller documents might come after
                continue

            if score < config.SCORE_THRESHOLD:
                if self.verbose:
                    print(
                        f"    - Skipped (low score) | Score: {score:.4f} Tokens: {tokens} | {doc.metadata['source']}"
                    )
                # We could break the loop as the list is sorted, but for output clarity we keep it
                continue

            selected_docs.append(doc)
            # Add also separator tokens count, as adding a document after will always require a separator
            current_tokens += tokens + self.doc_separator_tokens
            included_contents.add(content)
            if self.verbose:
                print(
                    f"    + Selected (Vector) | Score: {score:.4f} Tokens: {tokens} | {doc.metadata['source']}"
                )

        return selected_docs

    def answer_question(self, question):
        """
        Generates an answer for a single question.
        """
        if self.verbose:
            print(f"\n\n{'=' * 100}")
            print(f"Processing question: {question}\n")

        # 1. Retrieve docs
        selected_docs = self.retrieve_context(question)

        # 2. Format context
        context_text = config.DOC_SEPARATOR.join(
            [doc.page_content for doc in selected_docs]
        )
        if self.verbose:
            print(
                f"Total context tokens: {self.llm.get_num_tokens(context_text)}/{config.MAX_TOKENS}"
            )

        # 3. Generate
        message = self.prompt.format(context=context_text, question=question)
        response = self.llm.invoke(message)

        if self.verbose:
            print(f"ANSWER:\n{'-' * 100}\n{response.content}\n{'-' * 100}")
        else:
            print(f"\n{response.content}\n")

        return {
            "answer": response.content,
            "context": [doc.metadata["source"] for doc in selected_docs],
        }

    def run_batch(self, input_file, output_file):
        """
        Runs the pipeline on a JSON file containing a list of questions.
        """
        if self.verbose:
            print(f"Loading questions from {input_file}...")
        with open(input_file, "r") as f:
            questions_data = json.load(f)

        results = []
        for q_item in questions_data["questions"]:
            output = self.answer_question(q_item["question"])

            results.append(
                {
                    "id": q_item["id"],
                    "question": q_item["question"],
                    "answer": output["answer"],
                    "context": output["context"],
                }
            )

        with open(output_file, "w") as f:
            json.dump({"answers": results}, f, indent=2)

        if self.verbose:
            print(f"Answers saved to {output_file}")


if __name__ == "__main__":
    pipeline = RAGPipeline(verbose=True)
    pipeline.run_batch(config.QUESTIONS_FILE, config.RESULTS_FILE)
