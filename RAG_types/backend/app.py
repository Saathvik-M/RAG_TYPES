import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.chat import router as chat_router
from routers.upload import router as upload_router
from routers.health import router as health_router
from config import Config

app = FastAPI(
    title="RAG Playground",
    version="1.0"
)


@app.on_event("startup")
def validate_config():
    """Validate required configuration on startup."""
    if not Config.GROQ_API_KEY:
        print("=" * 60, file=sys.stderr)
        print("ERROR: GROQ_API_KEY is not set in backend/.env", file=sys.stderr)
        print("", file=sys.stderr)
        print("To get a FREE Groq API key:", file=sys.stderr)
        print("  1. Go to https://console.groq.com", file=sys.stderr)
        print("  2. Sign up (no credit card required)", file=sys.stderr)
        print("  3. Create an API key", file=sys.stderr)
        print("  4. Add it to backend/.env as:", file=sys.stderr)
        print("     GROQ_API_KEY=gsk_your_key_here", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
    else:
        print(f"✓ GROQ_API_KEY is configured (using model: {Config.GROQ_MODEL})")

# Allow CORS for Streamlit front‑end (default localhost:8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to "http://localhost:8501" if preferred
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(upload_router)
app.include_router(health_router)
