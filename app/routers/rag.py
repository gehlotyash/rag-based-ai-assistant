from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
)
from app.services.document_loader import load_and_chunk
from app.services.embedder import embed_and_store
from app.services.retriever import retrieve_chunks
from app.services.generator import generate_answer

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest):
    try:
        chunks = load_and_chunk(request.file_path)
        count = embed_and_store(chunks)
        return IngestResponse(message="Document ingested", chunks_created=count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    try:
        results = retrieve_chunks(request.question, request.top_k)
        chunks = [chunk for chunk, _ in results]
        sources = [source for _, source in results]
        answer = generate_answer(request.question, chunks)
        return QueryResponse(answer=answer, sources=sources, retrieved_chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
