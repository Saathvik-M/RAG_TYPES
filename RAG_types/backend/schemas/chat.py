from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    rag_type: str = "naive"


class Chunk(BaseModel):

    text: str

    page: int

    source: str


class ChatResponse(BaseModel):

    answer: str

    chunks: list[Chunk]

    rag: str

    model: str

    time_taken: float
