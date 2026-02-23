from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.api.routes_chat import router as chat_router
from app.api.routes_rag import router as rag_router

app = FastAPI(title="LLM Bootcamp (Ollama)")

app.include_router(chat_router)
app.include_router(rag_router)

@app.get("/health")
def health():
    return {"ok": True}
