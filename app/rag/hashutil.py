import xxhash

def hash_text(text: str) -> str:
    return xxhash.xxh3_128_hexdigest(text.strip())