import time
from typing import Generator, Any

from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError

from app.config import settings


client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url=settings.OPENAI_BASE_URL,  # points to Ollama
)


def chat_once(system_prompt: str, user_message: str, temperature: float) -> dict[str, Any]:
    """
    Non-streaming chat call.
    Works with Ollama via OpenAI-compatible endpoint.
    """
    start = time.time()

    try:
        resp = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        elapsed_ms = int((time.time() - start) * 1000)

        # Some local servers may omit usage. Handle gracefully.
        usage = getattr(resp, "usage", None)
        usage_dict = usage.model_dump() if usage else {}

        text = resp.choices[0].message.content or ""

        return {
            "text": text,
            "model": settings.OPENAI_MODEL,
            "latency_ms": elapsed_ms,
            "usage": usage_dict,
        }

    except RateLimitError as e:
        # Unlikely with local, but keep it professional.
        return {"error": "rate_limited", "detail": str(e)}
    except APIConnectionError as e:
        return {
            "error": "connection_error",
            "detail": "Cannot connect to Ollama. Is `brew services start ollama` running?",
            "raw": str(e),
        }
    except APIStatusError as e:
        return {"error": "api_status_error", "status_code": e.status_code, "detail": str(e)}
    except Exception as e:
        return {"error": "unknown_error", "detail": str(e)}


def chat_stream(system_prompt: str, user_message: str, temperature: float) -> Generator[str, None, None]:
    """
    Streaming chat call.
    Yields text chunks as they arrive.
    """
    try:
        stream = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=temperature,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )

        for event in stream:
            # OpenAI SDK stream events have `choices[0].delta.content`
            delta = event.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content

    except APIConnectionError:
        yield "\n[ERROR] Cannot connect to Ollama. Start it with: brew services start ollama\n"
    except Exception as e:
        yield f"\n[ERROR] {e}\n"
