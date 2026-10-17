import os
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)



class Config:

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

    GROQ_MODEL = "llama-3.1-8b-instant"

    CHUNK_SIZE = 1000

    CHUNK_OVERLAP = 200

    TOP_K = 3
