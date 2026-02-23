def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    i = 0
    n = len(text)

    while i < n:
        end = min(n, i + chunk_size)
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        i = max(0, end - overlap)

    return chunks