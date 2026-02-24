import os
from sqlalchemy import Column, Integer, Text, String, DateTime, func, Index, JSON
from pgvector.sqlalchemy import Vector
from app.db import Base

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)

    # identity / grouping
    namespace = Column(String, index=True, nullable=False, default="default")
    doc_id = Column(String, index=True, nullable=False)
    chunk_index = Column(Integer, nullable=False)

    # content
    content = Column(Text, nullable=False)

    # metadata
    title = Column(String, nullable=True)
    source = Column(String, nullable=True)   # url / filename
    tags = Column(JSON, nullable=True)       # ["hr", "policy"] etc.

    # caching keys
    content_hash = Column(String, index=True, nullable=False)

    # vector
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

Index("ix_chunks_ns_doc_chunk", Chunk.namespace, Chunk.doc_id, Chunk.chunk_index)
Index("ix_chunks_ns_hash", Chunk.namespace, Chunk.content_hash)