import time
import xxhash

# super-simple in-memory cache (good for dev)
# later you can move to Redis.
_ANSWER_CACHE: dict[str, tuple[float, dict]] = {}

def make_key(model: str, question: str, chunk_ids: list[int], temperature: float) -> str:
    raw = f"{model}|{temperature}|{question}|{','.join(map(str, chunk_ids))}"
    return xxhash.xxh3_128_hexdigest(raw)

def get_cached(key: str, ttl_sec: int = 300):
    item = _ANSWER_CACHE.get(key)
    if not item:
        return None
    ts, val = item
    if time.time() - ts > ttl_sec:
        _ANSWER_CACHE.pop(key, None)
        return None
    return val

def set_cached(key: str, value: dict):
    _ANSWER_CACHE[key] = (time.time(), value)