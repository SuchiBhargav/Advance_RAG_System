"""
Rate limiting implementation to prevent API abuse.
Uses sliding window algorithm with Redis backend.
"""

from typing import Dict
import time
from collections import defaultdict, deque
from fastapi import HTTPException, status

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window algorithm.
    For production, consider using Redis-based rate limiting.
    """
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
        
        logger.info(
            f"Initialized RateLimiter: {max_requests} requests per {window_seconds}s"
        )
    
    def check_rate_limit(self, client_id: str = "default") -> None:
        """
        Check if request is within rate limit.
        
        Args:
            client_id: Identifier for the client (IP, user ID, etc.)
        
        Raises:
            HTTPException: If rate limit exceeded
        """
        current_time = time.time()
        
        # Get request history for this client
        request_times = self.requests[client_id]
        
        # Remove old requests outside the window
        while request_times and request_times[0] < current_time - self.window_seconds:
            request_times.popleft()
        
        # Check if limit exceeded
        if len(request_times) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s"
            )
        
        # Add current request
        request_times.append(current_time)
        
        logger.debug(
            f"Rate limit check passed for {client_id}: "
            f"{len(request_times)}/{self.max_requests} requests"
        )

# Made with Bob
