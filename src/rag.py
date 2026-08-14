"""
RAG Engine - Retrieval-Augmented Generation Logic
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
import os


class RAGEngine:
    """RAG engine for document retrieval and question answering"""

    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
            temperature=0.7
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 50))
        )

    def add_document(self, text: str, metadata: dict = None):
        """Add a document to the vector store"""
        chunks = self.text_splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)

            self.vector_store.add_texts(
                texts=[chunk],
                metadatas=[chunk_metadata]
            )

    def query(self, question: str, k: int = 3, max_tokens: int = 500):
        """Query the RAG system"""
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(question, k=k)

        if not docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": []
            }

        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say so."),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])

        # Generate response
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "question": question
        })

        # Extract sources
        sources = list(set([
            doc.metadata.get("source", "unknown")
            for doc in docs
        ]))

        return {
            "answer": response.content,
            "sources": sources,
            "retrieved_chunks": len(docs)
        }

    def get_document_count(self) -> int:
        """Get total number of documents in the vector store"""
        return len(self.vector_store._collection.get()["ids"])
