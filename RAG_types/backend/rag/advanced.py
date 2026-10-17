import time

from services.retriever import Retriever
from services.prompt import PromptBuilder
from services.llm import GroqService
from rag.base import BaseRAG


start = time.time()


class AdvancedRAG(BaseRAG):
    """Advanced RAG with enhanced retrieval (e.g., MMR for diversity)."""

    def __init__(self):
        self.retriever = Retriever()
        self.llm = GroqService()

    def ask(self, question):
        docs = self.retriever.retrieve(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        prompt = PromptBuilder.build(
            context=context,
            question=question
        )

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
            "rag": "advanced",
            "time_taken": elapsed
        }