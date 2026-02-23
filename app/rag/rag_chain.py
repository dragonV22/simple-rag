from app.llm.client import chat_once

DEFAULT_RAG_SYSTEM = (
    "You are a helpful assistant. Answer using ONLY the provided CONTEXT.\n"
    "If the answer is not in the context, say: \"I don't know based on the provided context.\" \n"
    "Always include citations as chunk ids like [chunk:12]."
)

def answer_with_rag(question: str, retrieved_chunks, temperature: float = 0.2):
    context_lines = []
    for ch in retrieved_chunks:
        context_lines.append(f"[chunk:{ch.id}] {ch.content}")

    context = "\n\n".join(context_lines)

    user_message = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Return format:\n"
        "Answer: ...\n"
        "Citations: [chunk:<id>, ...]\n"
    )

    return chat_once(DEFAULT_RAG_SYSTEM, user_message, temperature)