from rag.naive import NaiveRAG
from rag.advanced import AdvancedRAG
from rag.hybrid import HybridRAG
from rag.graph import GraphRAG
from rag.agentic import AgenticRAG
from rag.self_rag import SelfRAG


class RAGFactory:

    _registry = {
        "naive": NaiveRAG,
        "advanced": AdvancedRAG,
        "hybrid": HybridRAG,
        "graph": GraphRAG,
        "agentic": AgenticRAG,
        "self": SelfRAG,
    }

    @classmethod
    def create(cls, rag_type: str):

        rag_class = cls._registry.get(rag_type.lower())

        if rag_class is None:
            raise ValueError(f"Unsupported RAG type: {rag_type}")

        return rag_class()