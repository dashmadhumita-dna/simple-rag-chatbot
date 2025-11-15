"""
Vector Store Setup - Initialize Chroma with OpenAI embeddings
"""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os


def setup_vector_store() -> Chroma:
    """
    Initialize Chroma vector store with OpenAI embeddings

    Returns:
        Chroma: Initialized vector store
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )

    # Create in-memory Chroma vector store
    vector_store = Chroma(
        embedding_function=embeddings,
        collection_name="rag_documents"
    )

    return vector_store


def get_embedding_model_info() -> dict:
    """Get information about the embedding model"""
    return {
        "model": "text-embedding-ada-002",
        "dimensions": 1536,
        "max_tokens": 8191
    }
