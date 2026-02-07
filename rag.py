import os
import logging
import numpy as np
import requests
import time
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, SecretStr
import chromadb
from chromadb.api.models.Collection import Collection
from langchain_openai import ChatOpenAI
import tiktoken


# Set offline mode for HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RAG_Chat_API")

# Configuration
FAST_API_ENDPOINT = os.getenv("FAST_API_ENDPOINT", "http://107.99.236.204:8080")
CHROMA_DEPLOYMENT = os.getenv("CHROMA_DEPLOYMENT", "local").lower()
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_collection")

# Initialize Chroma client (local by default, optional remote HTTP mode)
if CHROMA_DEPLOYMENT == "http":
    logger.info(f"Initializing Chroma HTTP client with host={CHROMA_HOST}, port={CHROMA_PORT}")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
else:
    logger.info(f"Initializing Chroma local persistent client with path={CHROMA_PERSIST_DIR}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    question: str

class Chunk(BaseModel):
    id: str
    source: str
    file_name: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    chunks: List[Chunk]

# Custom embedding class
class CustomEmbedding:
    def __init__(self, endpoint: str = FAST_API_ENDPOINT):
        self.endpoint = endpoint
        logger.info(f"CustomEmbedding initialized with endpoint: {endpoint}")
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.info(f"Embedding {len(texts)} documents")
        try:
            response = requests.post(f"{self.endpoint}/api/embed", json={"input": texts, "model": "qwen3-embedding:0.6b"})
            logger.info(f"Embedding service response status: {response.status_code}")
            if response.status_code == 200:
                embeddings = response.json()['embeddings']
                # Ensure embeddings is a list of lists - strength reduction
                if isinstance(embeddings, list):
                    if embeddings and not isinstance(embeddings[0], list):
                        embeddings = [embeddings]
                    logger.info(f"Successfully embedded {len(embeddings)} documents")
                    return embeddings
                else:
                    logger.error(f"Unexpected response format: {embeddings}")
                    raise ValueError("Embedding service response format not recognized")
            else:
                logger.error(f"Embedding service error: {response.status_code} - {response.text}")
                raise Exception(f"Embedding service error: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise e
    
    def embed_query(self, text: str) -> list[float]:
        logger.info(f"Embedding query: {text[:50]}...")
        # Function inlining - directly calling embed_documents instead of separate method
        try:
            embedding = self.embed_documents([text])[0]
            logger.info("Query embedding successful")
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise e

# Hybrid retriever class
class HybridRetriever:
    def __init__(self, chroma_client, collection_name: str):
        self.chroma = chroma_client
        self.collection_name = collection_name
        self.collection: Collection = self.chroma.get_or_create_collection(name=collection_name)
        # Initialize embedding models for querying
        try:
            self.dense_embedding_model = CustomEmbedding()
            logger.info("loaded dense")
            logger.info("Query models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing query models: {str(e)}")
            raise e
        logger.info(f"HybridRetriever initialized for collection: {collection_name}")
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        logger.info(f"Retrieving documents for query: {query[:50]}... (top_k={top_k})")
        try:
            # Generate dense query embedding
            logger.info("Generating query embeddings")
            dense_vectors = self.dense_embedding_model.embed_query(query)
            
            # Search Chroma using the query embedding
            logger.info("Performing Chroma vector search")
            search_result = self.collection.query(
                query_embeddings=[dense_vectors],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
            
            results = []
            if search_result and search_result.get("ids"):
                ids = search_result.get("ids", [[]])[0]
                metadatas = search_result.get("metadatas", [[]])[0]
                documents = search_result.get("documents", [[]])[0]
                distances = search_result.get("distances", [[]])[0]
                for idx, result_id in enumerate(ids):
                    metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
                    distance = distances[idx] if idx < len(distances) else 0.0
                    results.append({
                        "id": result_id,
                        "text": documents[idx] if idx < len(documents) else "",
                        "source": metadata.get("source", "") if metadata else "",
                        "file_name": metadata.get("file_name", "") if metadata else "",
                        "score": 1 - float(distance),
                    })
            
            logger.info(f"Retrieval complete. Returning top {min(top_k, len(results))} results")
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise e

# RAG generator class
class RAGGenerator:
    def __init__(self, model: str = "qwen3:0.6b"):
        logger.info(f"Initializing RAGGenerator with model: {model}")
        self.model = ChatOpenAI(
            model=model,
            base_url=os.getenv("LLM_BASE_URL", "http://107.97.203.41:30080/v1"),
            api_key=SecretStr(os.getenv("LLM_API_KEY", "nest2025")),
            temperature=0.1,
            top_p=0.9
        )
        self.max_context_tokens = 32000
        logger.info("RAGGenerator initialized")
        
    def generate(self, query: str, retrieved_docs: List[Dict]) -> Dict:
        logger.info(f"Generating response for query: {query[:50]}...")
        logger.info(f"Number of retrieved documents: {len(retrieved_docs)}")
        
        try:
            context = self._pack_context(retrieved_docs, self.max_context_tokens)
            logger.info(f"Packed context with {len(context)} documents")
            
            prompt = self._build_prompt(query, context)
            logger.debug(f"Prompt length: {len(prompt)} characters")
            
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            logger.info("Calling LLM for response generation")
            response = self.model.invoke(messages)
            logger.info("LLM response received")
            
            # Function inlining - directly create result dict
            result = {
                "answer": response.content,
                "sources": [doc.get("source", "unknown") for doc in retrieved_docs],
                "context_used": context
            }
            logger.info("Response generation complete")
            return result
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise e
    
    def _pack_context(self, docs: List[Dict], max_tokens: int) -> List[Dict]:
        logger.info(f"Packing context with max_tokens: {max_tokens}")
        # Sort docs by score in descending order
        sorted_docs = sorted(docs, key=lambda x: x.get('score', 0), reverse=True)
        logger.debug(f"Sorted documents by score")
        packed = []
        token_count = 0
        encoder = tiktoken.get_encoding("cl100k_base")
        logger.debug("Tokenizing documents")
        
        # Strength reduction - using enumerate instead of manual indexing
        for i, doc in enumerate(sorted_docs):
            doc_tokens = len(encoder.encode(doc['text']))
            logger.debug(f"Document {i+1} has {doc_tokens} tokens")
            
            if token_count + doc_tokens > max_tokens:
                logger.info(f"Token limit reached. Stopping at document {i+1}")
                break
                
            packed.append(doc)
            token_count += doc_tokens
            logger.debug(f"Added document {i+1} to context. Total tokens: {token_count}")
        
        logger.info(f"Context packing complete. Packed {len(packed)} documents with {token_count} tokens")
        return packed
    
    def _build_prompt(self, query: str, context: List[Dict]) -> str:
        logger.info("Building prompt")
        # Variable propagation - directly create context_text
        context_text = "\n\n".join([
            f"{i+1}. (Source: {doc.get('source', 'unknown')}, File Name: {doc.get('file_name', 'unknown')})\n{doc['text']}"
            for i, doc in enumerate(context)
        ])
        logger.info(f"Source {context[0].get('source')}: File Name  {context[0].get('file_name')}")
        logger.debug(f"Context text length: {len(context_text)} characters")
        
        # Constant folding - directly return the prompt string
        prompt = f"""You are provided with relevant document excerpts to answer the user's question. Your task is to provide a comprehensive, accurate, and helpful response based ONLY on the information contained in these documents.

Context Documents:
{context_text}

Question: {query}

Instructions:
1. Carefully analyze the provided documents to find information that answers the question
2. Provide a clear, comprehensive answer based solely on the context above
3. If the documents contain conflicting information, note this and explain the discrepancy
4. If the documents don't contain sufficient information to fully answer the question, clearly state what information is missing
5. Always cite your sources using the format Source after each claim or statement
6. When referring to specific files, include the file name in your citations
7. Organize your response logically with clear paragraphs and, if appropriate, bullet points or numbered lists
8. Use professional, precise language appropriate for a business or academic setting

Please provide your answer below:"""
        logger.info("Prompt built successfully")
        return prompt
    
    def _get_system_prompt(self) -> str:
        logger.info("Getting system prompt")
        # Constant folding - directly return the system prompt
        return """You are an expert document analysis assistant with the following capabilities and constraints:

Primary Role:
- Analyze provided documents to answer user questions accurately and comprehensively
- Extract, synthesize, and present information from multiple document sources
- Identify relationships, patterns, and key insights in the provided context

Strict Rules:
1. Use ONLY the information provided in the context documents
2. Cite every factual claim with the appropriate source using Source format
3. If information isn't available in the context, explicitly state "The provided documents don't contain information about [specific topic]"
4. Never make assumptions, inferences, or use external knowledge not present in the documents
5. Maintain professional, precise, and clear language throughout your responses
6. When documents contain conflicting information, acknowledge and explain the discrepancy
7. Organize responses logically with clear structure (paragraphs, bullet points, or numbered lists as appropriate)

Response Quality:
- Provide comprehensive answers that address all aspects of the question
- Use clear, concise language appropriate for professional or academic settings
- Ensure accuracy by strictly adhering to the provided context
- Structure responses for easy readability and understanding

Remember: Your credibility depends on strict adherence to these rules. When in doubt, cite your sources or acknowledge limitations."""

# Initialize components
logger.info("Initializing RAG components")
retriever = HybridRetriever(chroma_client, COLLECTION_NAME)
generator = RAGGenerator()
logger.info("RAG components initialized")

# FastAPI app
app = FastAPI(title="RAG Chat API", version="1.0.0")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that receives a query and returns an answer with relevant chunks.
    
    Args:
        request: ChatRequest containing the user's query
        
    Returns:
        ChatResponse containing the answer and relevant chunks
    """
    try:
        
        logger.info(f"Processing chat request with query: {request.question[:50]}...")
        
        # Retrieve relevant documents
        logger.info("Retrieving relevant documents...")
        retrieved_docs = retriever.retrieve(request.question, top_k=10)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Generate response
        logger.info("Generating response...")
        response = generator.generate(request.question, retrieved_docs)
        logger.info("Response generated successfully")
        
        # Convert retrieved documents to Chunk objects
        chunks = [
            Chunk(
                id=doc["id"],
                source=doc["source"],
                file_name=doc["file_name"],
                score=doc["score"]
            )
            for doc in retrieved_docs
        ]
        
        # Return response
        chat_response = ChatResponse(
            answer=response["answer"],
            chunks=chunks
        )
        
        logger.info("Chat request processed successfully")
        return chat_response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT_RAG_SYSTEM", "8001"))
    uvicorn.run("rag:app", host="0.0.0.0", port=port)

