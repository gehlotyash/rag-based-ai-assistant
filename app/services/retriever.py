from app.services.embedder import load_vectorstore
from typing import List, Tuple


def retrieve_chunks(question: str, top_k: int = 3) -> List[Tuple[str, str]]:
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search_with_score(question, k=top_k)
    return [
        (doc.page_content, doc.metadata.get("source", "unknown"))
        for doc, score in results
    ]
