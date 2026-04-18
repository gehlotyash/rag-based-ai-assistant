from fastapi import FastAPI
from app.routers import rag

app = FastAPI(title="RAG-Based AI Assistant", version="1.0.0")
app.include_router(rag.router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "RAG AI Assistant"}
