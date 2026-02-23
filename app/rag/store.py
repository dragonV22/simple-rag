from sqlalchemy.orm import Session
from sqlalchemy import select, delete
from app.rag.models import Chunk
from app.rag.embeddings import embed_texts

def upsert_document(session: Session, doc_id: str, chunks: list[str]) -> int:
    # Simple approach: replace doc chunks
    session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))

    embs = embed_texts(chunks)
    for idx, (c, e) in enumerate(zip(chunks, embs)):
        session.add(Chunk(doc_id=doc_id, chunk_index=idx, content=c, embedding=e))

    session.commit()
    return len(chunks)

def retrieve_top_k(session: Session, question: str, k: int = 5) -> list[Chunk]:
    q_emb = embed_texts([question])[0]

    # cosine distance works well for embeddings
    stmt = (
        select(Chunk)
        .order_by(Chunk.embedding.cosine_distance(q_emb))
        .limit(k)
    )
    return list(session.execute(stmt).scalars().all())