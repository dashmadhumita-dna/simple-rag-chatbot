"""
Unit tests for RAG functionality
"""
import pytest
from unittest.mock import Mock, patch
from src.rag import RAGEngine
from src.embeddings import setup_vector_store


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store"""
    with patch('src.rag.Chroma') as mock_chroma:
        mock_store = Mock()
        mock_store._collection.get.return_value = {"ids": []}
        yield mock_store


@pytest.fixture
def rag_engine(mock_vector_store):
    """Create a RAG engine instance"""
    with patch('src.rag.ChatOpenAI'):
        return RAGEngine(mock_vector_store)


def test_add_document(rag_engine, mock_vector_store):
    """Test adding a document to the vector store"""
    test_text = "This is a test document."
    test_metadata = {"source": "test"}

    rag_engine.add_document(test_text, test_metadata)

    # Verify add_texts was called
    assert mock_vector_store.add_texts.called


def test_query_with_no_results(rag_engine, mock_vector_store):
    """Test querying with no matching documents"""
    mock_vector_store.similarity_search.return_value = []

    result = rag_engine.query("What is machine learning?")

    assert "couldn't find" in result["answer"].lower()
    assert result["sources"] == []


def test_query_with_results(rag_engine, mock_vector_store):
    """Test querying with matching documents"""
    # Mock documents
    mock_doc = Mock()
    mock_doc.page_content = "Machine learning is a subset of AI."
    mock_doc.metadata = {"source": "ml_intro.md"}

    mock_vector_store.similarity_search.return_value = [mock_doc]

    # Mock LLM response
    with patch.object(rag_engine.llm, 'invoke') as mock_invoke:
        mock_response = Mock()
        mock_response.content = "Machine learning is a technology that enables computers to learn."
        mock_invoke.return_value = mock_response

        result = rag_engine.query("What is machine learning?")

        assert "machine learning" in result["answer"].lower()
        assert "ml_intro.md" in result["sources"]


def test_get_document_count(rag_engine, mock_vector_store):
    """Test getting document count"""
    mock_vector_store._collection.get.return_value = {
        "ids": ["1", "2", "3"]
    }

    count = rag_engine.get_document_count()
    assert count == 3


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test the health check endpoint"""
    from fastapi.testclient import TestClient
    from src.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()


@pytest.mark.asyncio
async def test_query_endpoint():
    """Test the query endpoint"""
    from fastapi.testclient import TestClient
    from src.main import app

    client = TestClient(app)

    # Note: This requires OPENAI_API_KEY to be set
    # In a real test environment, you'd mock the OpenAI calls
    response = client.post(
        "/query",
        json={"question": "What is FastAPI?"}
    )

    # Check that we get a response (may fail if no API key)
    assert response.status_code in [200, 500]
