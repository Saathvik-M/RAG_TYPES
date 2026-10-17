import os
from services.pdf_loader import PDFLoader

loader = PDFLoader()

base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_dir, "documents", "AI_Viva_Questions.pdf")
docs = loader.load_pdf(pdf_path)

print(f"Pages Loaded: {len(docs)}")

print()

print(docs[0].page_content[:500])