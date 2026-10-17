from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingService:

    _embedding_model = None

    def __init__(self):

        if EmbeddingService._embedding_model is None:
            EmbeddingService._embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        self.embedding_model = EmbeddingService._embedding_model

    def get_embeddings(self):
        return self.embedding_model