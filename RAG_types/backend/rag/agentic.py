import time

from services.retriever import Retriever
from services.llm import GroqService
from rag.base import BaseRAG


start = time.time()


class AgenticRAG(BaseRAG):
    """Agentic RAG with multi-step reasoning."""

    def __init__(self):
        self.retriever = Retriever()
        self.llm = GroqService()

    def ask(self, question):
        # Step 1: Determine what to search for (query decomposition)
        planning_prompt = f"""
You are a research assistant. Break down the following question into
key search queries (one per line) that would help find relevant information.
Only output the queries, nothing else.

Question: {question}
"""
        search_queries_text = self.llm.generate(planning_prompt)
        search_queries = [q.strip() for q in search_queries_text.strip().split("\n") if q.strip()]

        # Step 2: Retrieve for each sub-query
        all_docs = []
        seen_contents = set()
        for sq in search_queries[:3]:  # limit to 3 sub-queries
            docs = self.retriever.retrieve(sq)
            for doc in docs:
                if doc.page_content not in seen_contents:
                    all_docs.append(doc)
                    seen_contents.add(doc.page_content)

        context = "\n\n".join(
            doc.page_content
            for doc in all_docs
        )

        # Step 3: Synthesize final answer
        synthesis_prompt = f"""
You are an AI research assistant. Based on the following retrieved information,
provide a comprehensive answer to the question.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = self.llm.generate(synthesis_prompt)

        elapsed = round(time.time() - start, 2)

        return {
            "answer": answer,
            "chunks": [
                {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "source": doc.metadata.get("source", "")
                }
                for doc in all_docs
            ],
            "model": "llama3-70b-8192",
            "rag": "agentic",
            "time_taken": elapsed
        }