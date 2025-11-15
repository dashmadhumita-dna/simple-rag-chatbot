# Simple RAG Chatbot

A lightweight Retrieval-Augmented Generation (RAG) chatbot built with FastAPI, LangChain, and OpenAI.

## Features

- 🤖 RAG-powered Q&A using OpenAI embeddings
- 🚀 FastAPI backend with async support
- 🐳 Docker support for easy deployment
- ✅ Unit tests with pytest
- 📚 In-memory vector store (Chroma)
- 🔍 Document chunking and retrieval

## Architecture

```
┌─────────────┐
│   FastAPI   │  ← REST API endpoints
└──────┬──────┘
       │
┌──────▼──────┐
│  LangChain  │  ← RAG orchestration
└──────┬──────┘
       │
┌──────▼──────┐
│   Chroma    │  ← Vector store
└─────────────┘
```

## Quick Start

### Using Docker

```bash
# Build and run
docker-compose up --build

# Test the API
curl http://localhost:8000/health
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run the server
uvicorn src.main:app --reload

# Run tests
pytest tests/
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Query the chatbot
```bash
POST /query
Content-Type: application/json

{
  "question": "What is machine learning?"
}
```

### Add documents
```bash
POST /documents
Content-Type: application/json

{
  "text": "Machine learning is a subset of AI...",
  "metadata": {"source": "intro.md"}
}
```

## Project Structure

```
simple-rag-chatbot/
├── src/
│   ├── main.py           # FastAPI application
│   ├── rag.py            # RAG logic
│   └── embeddings.py     # Vector store setup
├── tests/
│   └── test_rag.py       # Unit tests
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container definition
├── docker-compose.yml    # Docker orchestration
└── README.md             # This file
```

## Tech Stack

- **Framework**: FastAPI 0.104+
- **LLM**: OpenAI GPT-3.5/4
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Store**: ChromaDB (in-memory)
- **Testing**: pytest

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `MODEL_NAME` | OpenAI model name | No (default: gpt-3.5-turbo) |
| `CHUNK_SIZE` | Text chunk size | No (default: 500) |
| `CHUNK_OVERLAP` | Chunk overlap | No (default: 50) |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_rag.py::test_query_response
```

## Deployment

### Docker

```bash
docker build -t simple-rag-chatbot .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key simple-rag-chatbot
```

### Cloud Platforms

- **Vercel**: Deploy FastAPI with serverless functions
- **Railway**: One-click deployment with Docker
- **Render**: Auto-deploy from GitHub

## Future Improvements

- [ ] Persistent vector store (PostgreSQL + pgvector)
- [ ] Multiple document sources
- [ ] Streaming responses
- [ ] Rate limiting
- [ ] Authentication
- [ ] Monitoring and logging
- [ ] Multi-language support

## License

MIT
