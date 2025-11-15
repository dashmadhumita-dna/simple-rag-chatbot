"""
Simple RAG Chatbot - FastAPI Application
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from .rag import RAGEngine
from .embeddings import setup_vector_store

# Initialize FastAPI app
app = FastAPI(
    title="Simple RAG Chatbot",
    description="A lightweight RAG-powered Q&A system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine: Optional[RAGEngine] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine
    vector_store = setup_vector_store()
    rag_engine = RAGEngine(vector_store)

    # Load sample documents
    sample_docs = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Python is a popular programming language for ML due to libraries like scikit-learn, TensorFlow, and PyTorch.",
        "RAG (Retrieval-Augmented Generation) combines retrieval and generation for better AI responses.",
        "FastAPI is a modern web framework for building APIs with Python, offering high performance and easy async support.",
    ]

    for doc in sample_docs:
        rag_engine.add_document(doc, {"source": "sample_data"})


# Request/Response models
class QueryRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 500


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple RAG Chatbot API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
        "documents_loaded": len(rag_engine.vector_store._collection.get()["ids"]) if rag_engine else 0
    }


@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """Query the RAG chatbot"""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")

    try:
        result = rag_engine.query(request.question, max_tokens=request.max_tokens)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/documents")
async def add_document(request: DocumentRequest):
    """Add a document to the vector store"""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")

    try:
        rag_engine.add_document(request.text, request.metadata or {})
        return {"message": "Document added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get chatbot statistics"""
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")

    collection = rag_engine.vector_store._collection.get()

    return {
        "total_documents": len(collection["ids"]),
        "total_chunks": len(collection["ids"]),
        "model": os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
