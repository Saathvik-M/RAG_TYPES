import time

from services.retriever import Retriever
from services.prompt import PromptBuilder
from services.llm import GroqService
from rag.base import BaseRAG


start = time.time()


class SelfRAG(BaseRAG):
    """Self-correcting RAG that generates, critiques, and revises answers."""

    def __init__(self):
        self.retriever = Retriever()
        self.llm = GroqService()

    def ask(self, question):
        # Step 1: Initial retrieval
        docs = self.retriever.retrieve(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # Step 2: Generate initial answer
        initial_prompt = PromptBuilder.build(
            context=context,
            question=question
        )
        initial_answer = self.llm.generate(initial_prompt)

        # Step 3: Self-critique
        critique_prompt = f"""
You are a fact-checker. Review the following answer and context.
Identify any statements in the answer that are NOT supported by the context.
If everything is supported, say "The answer is fully supported by the context."
If there are unsupported claims, list them.

Context:
{context}

Answer:
{initial_answer}

Critique:
"""
        critique = self.llm.generate(critique_prompt)

        # Step 4: Revise if needed
        if "fully supported" in critique.lower():
            final_answer = initial_answer
        else:
            revision_prompt = f"""
You are an AI assistant. Based on the following critique, revise your answer
to ensure it is fully supported by the context. Remove any unsupported claims.

Original Answer:
{initial_answer}

Critique:
{critique}

Context:
{context}

Question:
{question}

Revised Answer:
"""
            final_answer = self.llm.generate(revision_prompt)

        elapsed = round(time.time() - start, 2)

        return {
            "answer": final_answer,
            "chunks": [
                {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "source": doc.metadata.get("source", "")
                }
                for doc in docs
            ],
            "model": "llama3-70b-8192",
            "rag": "self",
            "time_taken": elapsed
        }