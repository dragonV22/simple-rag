import re

def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """
    Simple but stronger than v1:
    - normalize
    - split by paragraphs
    - pack paragraphs into chunks up to max_chars
    """
    text = normalize_text(text)
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paras:
        if len(p) > max_chars:
            # hard split long paragraph
            i = 0
            while i < len(p):
                part = p[i:i+max_chars].strip()
                if part:
                    chunks.append(part)
                i = max(0, i + max_chars - overlap)
            continue

        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            flush()
            buf = p

    flush()
    return chunks