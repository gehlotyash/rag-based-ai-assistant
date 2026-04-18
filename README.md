# rag-based-ai-assistant
Production-oriented RAG system using embeddings, vector search, and FastAPI inference APIs for semantic search and context-aware LLM responses

# RAG-Based AI Assistant
Production-oriented Retrieval-Augmented Generation system built for 
semantic search and context-aware LLM responses across enterprise datasets.

## Overview
This system enables intelligent question-answering over custom knowledge 
bases using a multi-stage retrieval and generation pipeline.

## Architecture
Documents → Chunking → Embedding → Vector Store
↓
Query → Embedding → Semantic Retrieval → Prompt Engineering → LLM → Response

## Tech Stack
- **Backend:** Python, FastAPI
- **Embeddings:** OpenAI / HuggingFace Sentence Transformers
- **Vector Store:** FAISS / pgvector (PostgreSQL)
- **LLM Integration:** Anthropic Claude API / OpenAI API
- **Prompt Engineering:** Custom retrieval-augmented prompt templates
- **Deployment:** Docker, REST API

## Key Features
- Semantic search using cosine similarity over dense vector embeddings
- Context-aware LLM responses grounded in retrieved document chunks
- FastAPI-based inference API for scalable production deployment
- Optimised chunking and retrieval logic for enterprise datasets
- Evaluation framework for retrieval quality and response accuracy

## Project Status
Active development — core RAG pipeline complete, 
inference API and evaluation layer in progress.

## Setup
```bash
git clone https://github.com/gehlotyash/RAG-Based-AI-Assistant
cd RAG-Based-AI-Assistant
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
uvicorn app.main:app --reload
```

## Folder Structure
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── routers/
│   │   └── rag.py           # RAG inference routes
│   ├── services/
│   │   ├── embedder.py      # Embedding generation
│   │   ├── retriever.py     # Vector search logic
│   │   └── generator.py     # LLM response generation
│   └── models/
│       └── schemas.py       # Pydantic request/response models
├── data/
│   └── sample_docs/         # Sample documents for testing
├── tests/
│   └── test_rag_pipeline.py
├── .env.example
├── requirements.txt
└── Dockerfile
