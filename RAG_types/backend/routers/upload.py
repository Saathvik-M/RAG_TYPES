from fastapi import (
    APIRouter,
    UploadFile,
    File
)

import os
import shutil

from services.pdf_loader import PDFLoader

from services.text_splitter import TextSplitter
from services.vector_store.faiss_store import FAISSStore
from schemas.upload import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(

    file: UploadFile = File(...)

):

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, "documents")
    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, file.filename)

    with open(path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    loader = PDFLoader()

    docs = loader.load_pdf(path)

    splitter = TextSplitter()

    chunks = splitter.split(docs)

    store = FAISSStore()

    store.create_vector_store(chunks)

    return UploadResponse(
        message="PDF uploaded successfully",
        chunks=len(chunks)
    )
