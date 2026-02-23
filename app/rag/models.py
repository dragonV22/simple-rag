import os
from sqlalchemy import Column, Integer, Text, String, DateTime, func, Index
from pgvector.sqlalchemy import Vector
from app.db import Base

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    doc_id = Column(String, index=True, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)

    # pgvector column
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

# Optional index (pgvector performance). We'll also create the extension in DB.
# You can run this later manually if you want.
# Note: ivfflat requires ANALYZE and works best with larger data.
Index("ix_chunks_doc_id_chunk_index", Chunk.doc_id, Chunk.chunk_index)