from pydantic import BaseModel
from typing import List


class IngestRequest(BaseModel):
    file_path: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieved_chunks: List[str]


class IngestResponse(BaseModel):
    message: str
    chunks_created: int
