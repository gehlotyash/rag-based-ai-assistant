import anthropic
from app.config import settings
from typing import List

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)


def generate_answer(question: str, context_chunks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are a helpful assistant. Answer the question using
ONLY the context provided below. If the answer is not in the context,
say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""

    message = client.messages.create(
        model=settings.model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
