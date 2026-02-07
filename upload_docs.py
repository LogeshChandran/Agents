import os
import uuid
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Dict

import chromadb
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("upload_docs.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DOC_UPLOAD_API")


# Configuration (match rag.py behavior)
FAST_API_ENDPOINT = os.getenv("FAST_API_ENDPOINT", "http://107.99.236.204:8080")
CHROMA_DEPLOYMENT = os.getenv("CHROMA_DEPLOYMENT", "local").lower()
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_collection")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


if CHROMA_DEPLOYMENT == "http":
    logger.info(f"Initializing Chroma HTTP client with host={CHROMA_HOST}, port={CHROMA_PORT}")
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
else:
    logger.info(f"Initializing Chroma local persistent client with path={CHROMA_PERSIST_DIR}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


class UploadResponse(BaseModel):
    file_name: str
    chunks_indexed: int


class BulkUploadResponse(BaseModel):
    uploaded: List[UploadResponse]


class CustomEmbedding:
    def __init__(self, endpoint: str = FAST_API_ENDPOINT):
        self.endpoint = endpoint

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            f"{self.endpoint}/api/embed",
            json={"input": texts, "model": "qwen3-embedding:0.6b"},
            timeout=120,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Embedding service error: {response.status_code} - {response.text}")

        embeddings = response.json().get("embeddings", [])
        if not isinstance(embeddings, list):
            raise RuntimeError("Embedding service response format not recognized")

        if embeddings and not isinstance(embeddings[0], list):
            embeddings = [embeddings]

        return embeddings


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    chunks = []
    start = 0
    n = len(clean)

    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(clean[start:end])
        if end == n:
            break
        start = max(end - overlap, start + 1)

    return chunks


def extract_text_from_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader  # optional dependency
    except Exception as exc:
        raise RuntimeError("PDF support requires pypdf. Install with: pip install pypdf") from exc

    reader = PdfReader(BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_docx(content: bytes) -> str:
    try:
        from docx import Document  # optional dependency (python-docx)
    except Exception as exc:
        raise RuntimeError("DOCX support requires python-docx. Install with: pip install python-docx") from exc

    document = Document(BytesIO(content))
    return "\n".join([p.text for p in document.paragraphs])


def extract_text(file_name: str, content: bytes) -> str:
    suffix = Path(file_name).suffix.lower()

    if suffix in {".txt", ".md", ".csv", ".log"}:
        return content.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        return extract_text_from_pdf(content)

    if suffix == ".docx":
        return extract_text_from_docx(content)

    raise RuntimeError(f"Unsupported file type: {suffix}. Supported: .txt .md .csv .log .pdf .docx")


embedding_client = CustomEmbedding()
app = FastAPI(title="Document Upload API", version="1.0.0")


@app.post("/upload", response_model=BulkUploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    results: List[UploadResponse] = []

    for file in files:
        try:
            raw = await file.read()
            text = extract_text(file.filename, raw)
            chunks = chunk_text(text)

            if not chunks:
                logger.warning(f"No text extracted from file: {file.filename}")
                results.append(UploadResponse(file_name=file.filename, chunks_indexed=0))
                continue

            embeddings = embedding_client.embed_documents(chunks)
            if len(embeddings) != len(chunks):
                raise RuntimeError("Embedding count does not match chunk count")

            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas: List[Dict] = [
                {
                    "source": file.filename,
                    "file_name": file.filename,
                    "chunk_index": idx,
                }
                for idx in range(len(chunks))
            ]

            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
            )

            results.append(UploadResponse(file_name=file.filename, chunks_indexed=len(chunks)))
            logger.info(f"Indexed file={file.filename} chunks={len(chunks)}")

        except Exception as exc:
            logger.exception(f"Failed to process file={file.filename}: {exc}")
            raise HTTPException(status_code=400, detail=f"Failed for {file.filename}: {exc}") from exc

    return BulkUploadResponse(uploaded=results)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT_UPLOAD_DOCS", "8002"))
    uvicorn.run("upload_docs:app", host="0.0.0.0", port=port)
