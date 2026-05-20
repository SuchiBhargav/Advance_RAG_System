"""
Production-ready Redis caching service for RAG system.
Implements intelligent caching with TTL, compression, and cache warming.
"""

import json
import hashlib
import pickle
from typing import Optional, Any, Dict, cast
from redis import Redis
from redis.exceptions import RedisError
import zlib

from config.settings import settings
from src.utils.logger import get_logger
from src.models.schemas import QueryResponse

logger = get_logger(__name__)


class RedisCache:
    """
    Production-ready Redis cache with compression and error handling.
    """
    
    def __init__(self):
        """Initialize Redis connection with retry logic."""
        self.redis_client: Optional[Redis] = None
        self.enabled = False
        
        try:
            self.redis_client = Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=False,  # We'll handle encoding
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis cache connected: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            self.enabled = True
        except RedisError as e:
            logger.error(f"Redis connection failed: {e}. Cache disabled.")
            self.redis_client = None
            self.enabled = False
    
    def _generate_cache_key(self, query: str, filters: Optional[Dict] = None) -> str:
        """
        Generate consistent cache key from query and filters.
        
        Args:
            query: User query
            filters: Optional metadata filters
        
        Returns:
            Cache key string
        """
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Include filters in key if present
        key_data = {
            "query": normalized_query,
            "filters": filters or {}
        }
        
        # Create hash
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        return f"rag:query:{key_hash}"
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using zlib."""
        return zlib.compress(data, level=6)
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using zlib."""
        return zlib.decompress(data)
    
    def get(self, query: str, filters: Optional[Dict] = None) -> Optional[QueryResponse]:
        """
        Get cached response for a query.
        
        Args:
            query: User query
            filters: Optional metadata filters
        
        Returns:
            Cached QueryResponse or None if not found
        """
        if not self.enabled:
            return None
        
        try:
            if not self.redis_client:
                return None
            
            key = self._generate_cache_key(query, filters)
            cached_data = self.redis_client.get(key)
            
            if cached_data and isinstance(cached_data, bytes):
                # Decompress and deserialize
                decompressed = self._decompress_data(cached_data)
                response_dict = pickle.loads(decompressed)
                
                # Convert back to QueryResponse
                response = QueryResponse(**response_dict)
                
                logger.info(f"Cache HIT for query: {query[:50]}...")
                
                # Update metadata to indicate cache hit
                if response.metadata is None:
                    response.metadata = {}
                response.metadata["cache_hit"] = True
                
                return response
            
            logger.debug(f"Cache MISS for query: {query[:50]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    def set(
        self,
        query: str,
        response: QueryResponse,
        filters: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a query response.
        
        Args:
            query: User query
            response: QueryResponse to cache
            filters: Optional metadata filters
            ttl: Time to live in seconds (default from settings)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            if not self.redis_client:
                return False
            
            key = self._generate_cache_key(query, filters)
            ttl = ttl or settings.CACHE_TTL
            
            # Serialize and compress
            response_dict = response.model_dump()
            serialized = pickle.dumps(response_dict)
            compressed = self._compress_data(serialized)
            
            # Store with TTL
            self.redis_client.setex(key, ttl, compressed)
            
            logger.debug(f"Cached response for query: {query[:50]}... (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Error caching response: {e}")
            return False
    
    def invalidate(self, query: str, filters: Optional[Dict] = None) -> bool:
        """
        Invalidate a cached query.
        
        Args:
            query: User query
            filters: Optional metadata filters
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            if not self.redis_client:
                return False
            
            key = self._generate_cache_key(query, filters)
            deleted = self.redis_client.delete(key)
            
            if deleted:
                logger.info(f"Invalidated cache for query: {query[:50]}...")
            
            return bool(deleted)
            
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str = "rag:query:*") -> int:
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: Redis key pattern
        
        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0
        
        try:
            if not self.redis_client:
                return 0
            
            keys = self.redis_client.keys(pattern)
            if keys and isinstance(keys, list):
                deleted_count = self.redis_client.delete(*keys)
                # Cast to int since Redis returns int for delete operations
                deleted = cast(int, deleted_count) if deleted_count is not None else 0
                logger.info(f"Invalidated {deleted} cached queries")
                return deleted
            return 0
            
        except Exception as e:
            logger.error(f"Error invalidating pattern: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        if not self.enabled:
            return {"enabled": False}
        
        try:
            if not self.redis_client:
                return {"enabled": False}
            
            info_data = self.redis_client.info("stats")
            keys = self.redis_client.keys("rag:query:*")
            keys_count = len(keys) if isinstance(keys, list) else 0
            
            # Cast info to dict for type safety
            info_dict = cast(Dict[str, Any], info_data) if isinstance(info_data, dict) else {}
            
            return {
                "enabled": True,
                "total_keys": keys_count,
                "keyspace_hits": info_dict.get("keyspace_hits", 0),
                "keyspace_misses": info_dict.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info_dict.get("keyspace_hits", 0),
                    info_dict.get("keyspace_misses", 0)
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate."""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0
    
    def health_check(self) -> bool:
        """
        Check if Redis is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            if not self.redis_client:
                return False
            return bool(self.redis_client.ping())
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


# Global cache instance
_cache_instance: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """
    Get or create global cache instance.
    
    Returns:
        RedisCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


# Made with Bob