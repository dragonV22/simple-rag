from sqlalchemy.orm import Session
from sqlalchemy import select, delete, and_
from app.rag.models import Chunk
from app.rag.embeddings import embed_texts
from app.rag.hashutil import hash_text

def upsert_document(
    session: Session,
    namespace: str,
    doc_id: str,
    chunks: list[str],
    title: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
) -> int:
    # delete existing doc chunks
    session.execute(delete(Chunk).where(and_(Chunk.namespace == namespace, Chunk.doc_id == doc_id)))

    # compute hashes
    hashes = [hash_text(c) for c in chunks]

    # find which hashes already exist in this namespace (embedding reuse)
    existing = session.execute(
        select(Chunk.content_hash, Chunk.embedding).where(
            and_(Chunk.namespace == namespace, Chunk.content_hash.in_(hashes))
        )
    ).all()
    existing_map = {h: e for (h, e) in existing}

    # embed only missing
    missing_texts = [c for c, h in zip(chunks, hashes) if h not in existing_map]
    missing_embs = embed_texts(missing_texts) if missing_texts else []
    missing_iter = iter(missing_embs)

    for idx, (c, h) in enumerate(zip(chunks, hashes)):
        emb = existing_map.get(h)
        if emb is None:
            emb = next(missing_iter)

        session.add(
            Chunk(
                namespace=namespace,
                doc_id=doc_id,
                chunk_index=idx,
                content=c,
                title=title,
                source=source,
                tags=tags or [],
                content_hash=h,
                embedding=emb,
            )
        )

    session.commit()
    return len(chunks)

def retrieve_top_k(
    session: Session,
    namespace: str,
    question: str,
    k: int = 5,
    doc_ids: list[str] | None = None,
    tags_any: list[str] | None = None,
):
    q_emb = embed_texts([question])[0]

    stmt = select(Chunk).where(Chunk.namespace == namespace)

    if doc_ids:
        stmt = stmt.where(Chunk.doc_id.in_(doc_ids))

    # tags filter (simple): if tags_any, keep chunks whose tags overlap
    if tags_any:
        # JSON containment is DB-dependent; keep it simple in Week 3:
        # fetch more then filter in Python (good enough at small scale).
        stmt = stmt.order_by(Chunk.embedding.cosine_distance(q_emb)).limit(max(50, k * 10))
        rows = list(session.execute(stmt).scalars().all())
        filtered = [r for r in rows if r.tags and any(t in r.tags for t in tags_any)]
        return filtered[:k]

    stmt = stmt.order_by(Chunk.embedding.cosine_distance(q_emb)).limit(k)
    return list(session.execute(stmt).scalars().all())