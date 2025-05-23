import redis
import hashlib
# Connect to Redis
cache = redis.Redis(host='localhost', port=6379, db=0)

def get_cache_key(question: str) -> str:
    """Create a consistent key using a hash."""
    return "q:" + hashlib.sha256(question.lower().strip().encode()).hexdigest()

def get_cached_answer(question: str):
    key = get_cache_key(question)
    value = cache.get(key)
    if value:
        return value.decode()   
    return None

def set_cached_answer(question: str, answer: str):
    key = get_cache_key(question)
    cache.setex(key, 86400, answer)  # Cache for 1 day (optional TTL)
