from fastapi import APIRouter, Depends, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from pypdf import PdfReader
import io

from app.db import SessionLocal, engine
from app.rag.models import Chunk
from app.rag.chunking import chunk_text
from app.rag.store import upsert_document, retrieve_top_k
from app.rag.rag_chain import answer_with_rag

# Create tables
Chunk.metadata.create_all(bind=engine)

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class IngestTextRequest(BaseModel):
    doc_id: str = Field(..., description="Unique document id")
    text: str = Field(..., description="Raw text content")

class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

@router.post("/rag/ingest_text")
def ingest_text(req: IngestTextRequest, db: Session = Depends(get_db)):
    chunks = chunk_text(req.text)
    n = upsert_document(db, req.doc_id, chunks)
    return {"doc_id": req.doc_id, "chunks_ingested": n}

@router.post("/rag/ingest_pdf")
async def ingest_pdf(doc_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    data = await file.read()
    reader = PdfReader(io.BytesIO(data))
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    chunks = chunk_text(text)
    n = upsert_document(db, doc_id, chunks)
    return {"doc_id": doc_id, "chunks_ingested": n, "pages": len(reader.pages)}

@router.post("/rag/ask")
def rag_ask(req: AskRequest, db: Session = Depends(get_db)):
    hits = retrieve_top_k(db, req.question, req.top_k)
    out = answer_with_rag(req.question, hits, temperature=req.temperature)
    out["retrieved_chunk_ids"] = [h.id for h in hits]
    return out