import io
import uuid
from fastapi import APIRouter, Depends, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from pypdf import PdfReader

from app.db import SessionLocal, engine
from app.rag.models import Chunk
from app.rag.chunking import chunk_text
from app.rag.store import upsert_document, retrieve_top_k
from app.rag.rag_chain import answer_with_rag
from app.rag.cache import make_key, get_cached, set_cached
from app.config import settings

Chunk.metadata.create_all(bind=engine)

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class IngestTextRequest(BaseModel):
    namespace: str = "default"
    doc_id: str
    title: str | None = None
    source: str | None = None
    tags: list[str] = []
    text: str

class AskRequest(BaseModel):
    namespace: str = "default"
    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    # filters
    doc_ids: list[str] | None = None
    tags_any: list[str] | None = None

@router.post("/rag/ingest_text")
def ingest_text(req: IngestTextRequest, db: Session = Depends(get_db)):
    request_id = str(uuid.uuid4())
    chunks = chunk_text(req.text)
    n = upsert_document(
        db,
        namespace=req.namespace,
        doc_id=req.doc_id,
        chunks=chunks,
        title=req.title,
        source=req.source,
        tags=req.tags,
    )
    return {"request_id": request_id, "namespace": req.namespace, "doc_id": req.doc_id, "chunks_ingested": n}

@router.post("/rag/ingest_pdf")
async def ingest_pdf(
    namespace: str,
    doc_id: str,
    title: str | None = None,
    source: str | None = None,
    tags: str | None = None,   # comma-separated
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    request_id = str(uuid.uuid4())
    data = await file.read()
    reader = PdfReader(io.BytesIO(data))

    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    text = "\n\n".join(text_parts)

    chunks = chunk_text(text)
    tags_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
    n = upsert_document(
        db,
        namespace=namespace,
        doc_id=doc_id,
        chunks=chunks,
        title=title,
        source=source or file.filename,
        tags=tags_list,
    )
    return {"request_id": request_id, "namespace": namespace, "doc_id": doc_id, "chunks_ingested": n, "pages": len(reader.pages)}

@router.post("/rag/ask")
def rag_ask(req: AskRequest, db: Session = Depends(get_db)):
    request_id = str(uuid.uuid4())

    hits = retrieve_top_k(
        db,
        namespace=req.namespace,
        question=req.question,
        k=req.top_k,
        doc_ids=req.doc_ids,
        tags_any=req.tags_any,
    )

    chunk_ids = [h.id for h in hits]
    cache_key = make_key(settings.OPENAI_MODEL, req.question, chunk_ids, req.temperature)
    cached = get_cached(cache_key, ttl_sec=300)
    if cached:
        cached["request_id"] = request_id
        cached["cache_hit"] = True
        return cached

    out = answer_with_rag(req.question, hits, temperature=req.temperature)
    out["request_id"] = request_id
    out["cache_hit"] = False
    out["retrieved_chunk_ids"] = chunk_ids
    set_cached(cache_key, out)
    return out