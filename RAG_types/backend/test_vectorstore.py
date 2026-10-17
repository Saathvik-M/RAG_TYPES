import os
from services.pdf_loader import PDFLoader
from services.text_splitter import TextSplitter
from services.vector_store import VectorStore

loader = PDFLoader()

base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_dir, "documents", "AI_Viva_Questions.pdf")
documents = loader.load_pdf(pdf_path)

splitter = TextSplitter()

chunks = splitter.split(documents)

store = VectorStore()

store.create_vector_store(chunks)

print("Vector Store Created Successfully!")