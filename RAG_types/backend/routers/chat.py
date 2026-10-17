import traceback

from fastapi import APIRouter, HTTPException

from rag.factory import RAGFactory

from schemas.chat import (
    ChatRequest,
    ChatResponse
)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    try:
        rag = RAGFactory.create(request.rag_type)
        response = rag.ask(
            request.question
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_msg = f"Internal server error: {str(e)}"
        traceback.print_exc()
        print(f"ERROR: {error_msg}", flush=True)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

    return ChatResponse(
        answer=response["answer"],
        chunks=response["chunks"],
        rag=response.get("rag", request.rag_type),
        model=response.get("model", "llama3-70b-8192"),
        time_taken=response.get("time_taken", 0.0)
    )


@router.get("/rag-types")
def rag_types():

    return {
        "types":[
            "naive",
            "advanced",
            "hybrid",
            "graph",
            "agentic",
            "self"
        ]
    }
