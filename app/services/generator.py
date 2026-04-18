from transformers import pipeline
from typing import List
import logging

logger = logging.getLogger(__name__)

_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        logger.info("Loading local model...")
        _pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_new_tokens=256,
            do_sample=False,
        )
        logger.info("Local model loaded.")
    return _pipe


def generate_answer(question: str, context_chunks: List[str]) -> str:
    context = " ".join(context_chunks)
    prompt = f"Answer the question based only on this context.\nContext: {context}\nQuestion: {question}\nAnswer:"
    result = get_pipeline()(prompt)
    generated = result[0]["generated_text"]
    if "Answer:" in generated:
        return generated.split("Answer:")[-1].strip()
    return generated.strip()
