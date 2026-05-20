"""
Metrics collection service for monitoring system performance.
Tracks queries, response times, errors, and other key metrics.
"""

from typing import Dict, Any, List
from collections import defaultdict
import time
from datetime import datetime
from threading import Lock

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Collects and aggregates system metrics for monitoring.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = time.time()
        self.lock = Lock()
        
        # Metrics storage
        self.total_queries = 0
        self.total_errors = 0
        self.response_times: List[float] = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Query type distribution
        self.query_types: Dict[str, int] = defaultdict(int)
        
        # Confidence scores
        self.confidence_scores: List[float] = []
        
        # Feedback
        self.feedback_ratings: List[int] = []
        self.helpful_count = 0
        self.not_helpful_count = 0
        
        # Endpoint metrics
        self.endpoint_calls: Dict[str, int] = defaultdict(int)
        self.endpoint_errors: Dict[str, int] = defaultdict(int)
        
        logger.info("Initialized MetricsCollector")
    
    def record_query(
        self,
        query_type: str,
        confidence: float,
        processing_time: float
    ) -> None:
        """
        Record a query execution.
        
        Args:
            query_type: Type of query
            confidence: Confidence score
            processing_time: Processing time in milliseconds
        """
        with self.lock:
            self.total_queries += 1
            self.response_times.append(processing_time)
            self.query_types[query_type] += 1
            self.confidence_scores.append(confidence)
            
            logger.debug(
                f"Recorded query: type={query_type}, "
                f"confidence={confidence:.2f}, time={processing_time:.2f}ms"
            )
    
    def record_error(self, error_type: str = "unknown") -> None:
        """
        Record an error occurrence.
        
        Args:
            error_type: Type of error
        """
        with self.lock:
            self.total_errors += 1
            logger.debug(f"Recorded error: {error_type}")
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self.lock:
            self.cache_hits += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self.lock:
            self.cache_misses += 1
    
    def record_feedback(self, rating: int, is_helpful: bool) -> None:
        """
        Record user feedback.
        
        Args:
            rating: Rating from 1-5
            is_helpful: Whether response was helpful
        """
        with self.lock:
            self.feedback_ratings.append(rating)
            if is_helpful:
                self.helpful_count += 1
            else:
                self.not_helpful_count += 1
            
            logger.debug(f"Recorded feedback: rating={rating}, helpful={is_helpful}")
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float
    ) -> None:
        """
        Record an API request.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            status_code: Response status code
            duration: Request duration in seconds
        """
        with self.lock:
            key = f"{method} {endpoint}"
            self.endpoint_calls[key] += 1
            
            if status_code >= 400:
                self.endpoint_errors[key] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics snapshot.
        
        Returns:
            Dictionary of metrics
        """
        with self.lock:
            # Calculate averages
            avg_response_time = (
                sum(self.response_times) / len(self.response_times)
                if self.response_times else 0.0
            )
            
            avg_confidence = (
                sum(self.confidence_scores) / len(self.confidence_scores)
                if self.confidence_scores else 0.0
            )
            
            avg_rating = (
                sum(self.feedback_ratings) / len(self.feedback_ratings)
                if self.feedback_ratings else 0.0
            )
            
            # Calculate cache hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (
                self.cache_hits / total_cache_requests
                if total_cache_requests > 0 else 0.0
            )
            
            # Calculate error rate
            total_requests = self.total_queries + self.total_errors
            error_rate = (
                self.total_errors / total_requests
                if total_requests > 0 else 0.0
            )
            
            # Calculate uptime
            uptime = time.time() - self.start_time
            
            metrics = {
                "total_queries": self.total_queries,
                "total_errors": self.total_errors,
                "avg_response_time": avg_response_time,
                "avg_confidence": avg_confidence,
                "cache_hit_rate": cache_hit_rate,
                "error_rate": error_rate,
                "uptime": uptime,
                "query_types": dict(self.query_types),
                "feedback": {
                    "avg_rating": avg_rating,
                    "helpful_count": self.helpful_count,
                    "not_helpful_count": self.not_helpful_count,
                    "total_feedback": len(self.feedback_ratings)
                },
                "endpoints": {
                    "calls": dict(self.endpoint_calls),
                    "errors": dict(self.endpoint_errors)
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return metrics
    
    def reset_metrics(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self.lock:
            self.total_queries = 0
            self.total_errors = 0
            self.response_times.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.query_types.clear()
            self.confidence_scores.clear()
            self.feedback_ratings.clear()
            self.helpful_count = 0
            self.not_helpful_count = 0
            self.endpoint_calls.clear()
            self.endpoint_errors.clear()
            
            logger.info("Metrics reset")

# Made with Bob
