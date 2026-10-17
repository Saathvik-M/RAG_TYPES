import os
from services.pdf_loader import PDFLoader
from services.text_splitter import TextSplitter

loader = PDFLoader()

base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_dir, "documents", "AI_Viva_Questions.pdf")
documents = loader.load_pdf(pdf_path)

splitter = TextSplitter()

chunks = splitter.split(documents)

print(f"Total Chunks: {len(chunks)}")

print()

print(chunks[0].page_content)

print()

print(chunks[1].page_content)