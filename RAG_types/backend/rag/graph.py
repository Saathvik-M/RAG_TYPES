import time

from services.retriever import Retriever
from services.prompt import PromptBuilder
from services.llm import GroqService
from rag.base import BaseRAG


start = time.time()


class GraphRAG(BaseRAG):
    """Graph-based RAG that uses relationship-aware retrieval."""

    def __init__(self):
        self.retriever = Retriever()
        self.llm = GroqService()

    def ask(self, question):
        docs = self.retriever.retrieve(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # Graph-aware prompt
        prompt = f"""
You are an AI assistant with graph-based reasoning capabilities.

Analyze the following context and answer the question. If relationships exist
between different pieces of information, explain them clearly.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = self.llm.generate(prompt)

        elapsed = round(time.time() - start, 2)

        return {
            "answer": answer,
            "chunks": [
                {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "source": doc.metadata.get("source", "")
                }
                for doc in docs
            ],
            "model": "llama3-70b-8192",
            "rag": "graph",
            "time_taken": elapsed
        }