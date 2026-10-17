from services.vector_store.base import VectorStore


import os

class Retriever:

    def __init__(self):
        self._db = None

    def get_db(self):
        if self._db is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            store_dir = os.path.join(base_dir, "vectorstore")
            if not os.path.exists(store_dir):
                raise ValueError("No documents have been uploaded yet. Please upload a PDF first.")
            self._db = VectorStore().load_vector_store()
        return self._db

    def retrieve(self, question):
        db = self.get_db()
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        return retriever.invoke(question)