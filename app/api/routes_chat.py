from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.llm.client import chat_once, chat_stream
from app.llm.prompts import DEFAULT_SYSTEM_PROMPT


router = APIRouter()


class ChatRequest(BaseModel):
    system_prompt: str = Field(default=DEFAULT_SYSTEM_PROMPT)
    user_message: str
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


@router.post("/chat")
def post_chat(req: ChatRequest):
    return chat_once(
        system_prompt=req.system_prompt,
        user_message=req.user_message,
        temperature=req.temperature,
    )


@router.post("/chat/stream")
def post_chat_stream(req: ChatRequest):
    def generator():
        yield from chat_stream(
            system_prompt=req.system_prompt,
            user_message=req.user_message,
            temperature=req.temperature,
        )

    return StreamingResponse(generator(), media_type="text/plain; charset=utf-8")
