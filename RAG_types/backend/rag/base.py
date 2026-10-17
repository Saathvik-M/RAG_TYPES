from abc import ABC, abstractmethod


class BaseRAG(ABC):

    @abstractmethod
    def ask(self, question: str):
        pass