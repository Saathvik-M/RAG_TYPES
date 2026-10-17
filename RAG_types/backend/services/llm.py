from langchain_groq import ChatGroq

from config import Config


class GroqService:

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            if not Config.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in backend/.env. Please configure it first.")
            self._llm = ChatGroq(
                model=Config.GROQ_MODEL,
                api_key=Config.GROQ_API_KEY,
                temperature=0
            )
        return self._llm

    def generate(self, prompt):
        response = self.llm.invoke(prompt)
        return response.content