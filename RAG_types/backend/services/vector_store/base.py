import os
from langchain_community.vectorstores import FAISS

from services.embeddings import EmbeddingService


class VectorStore:

    def __init__(self):

        self.embeddings = EmbeddingService().get_embeddings()
        # Resolve backend root directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.store_dir = os.path.join(self.base_dir, "vectorstore")

    def create_vector_store(self, chunks):

        db = FAISS.from_documents(
            chunks,
            self.embeddings
        )

        db.save_local(self.store_dir)

        return db

    def load_vector_store(self):

        return FAISS.load_local(
            self.store_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
        )