class PromptBuilder:

    @staticmethod
    def build(context: str, question: str):

        return f"""
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not present, say:

"I couldn't find that information in the document."

Context:

{context}

Question:

{question}

Answer:
"""