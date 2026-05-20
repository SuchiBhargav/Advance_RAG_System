"""
Conversation memory management for multi-turn RAG conversations.
Implements session-based memory with context window management.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque
import json

from config.settings import settings
from src.utils.logger import get_logger
from src.services.redis_cache import get_cache

logger = get_logger(__name__)


class ConversationTurn:
    """Represents a single turn in a conversation."""
    
    def __init__(
        self,
        query: str,
        answer: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize conversation turn.
        
        Args:
            query: User query
            answer: System answer
            timestamp: Turn timestamp
            metadata: Additional metadata
        """
        self.query = query
        self.answer = answer
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create from dictionary."""
        return cls(
            query=data["query"],
            answer=data["answer"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


class ConversationMemory:
    """
    Manages conversation history with context window and persistence.
    """
    
    def __init__(
        self,
        session_id: str,
        max_turns: int = 10,
        context_window: int = 3,
        ttl_hours: int = 24
    ):
        """
        Initialize conversation memory.
        
        Args:
            session_id: Unique session identifier
            max_turns: Maximum turns to store
            context_window: Number of recent turns to include in context
            ttl_hours: Time to live for session in hours
        """
        self.session_id = session_id
        self.max_turns = max_turns
        self.context_window = context_window
        self.ttl_hours = ttl_hours
        self.turns: deque = deque(maxlen=max_turns)
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        
        # Try to load from cache
        self._load_from_cache()
        
        logger.info(f"Initialized ConversationMemory for session: {session_id}")
    
    def add_turn(self, query: str, answer: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a new conversation turn.
        
        Args:
            query: User query
            answer: System answer
            metadata: Optional metadata
        """
        turn = ConversationTurn(query, answer, metadata=metadata)
        self.turns.append(turn)
        self.last_accessed = datetime.utcnow()
        
        # Persist to cache
        self._save_to_cache()
        
        logger.debug(f"Added turn to session {self.session_id}: {query[:50]}...")
    
    def get_context(self, include_current: bool = False) -> str:
        """
        Get conversation context for the LLM.
        
        Args:
            include_current: Whether to include the current turn
        
        Returns:
            Formatted conversation context
        """
        if not self.turns:
            return ""
        
        # Get recent turns within context window
        recent_turns = list(self.turns)[-self.context_window:]
        
        if not include_current and len(recent_turns) > 0:
            recent_turns = recent_turns[:-1]
        
        if not recent_turns:
            return ""
        
        # Format context
        context_parts = ["Previous conversation:"]
        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(f"\nTurn {i}:")
            context_parts.append(f"User: {turn.query}")
            context_parts.append(f"Assistant: {turn.answer[:200]}...")  # Truncate long answers
        
        return "\n".join(context_parts)
    
    def get_recent_queries(self, n: int = 3) -> List[str]:
        """
        Get recent queries for context.
        
        Args:
            n: Number of recent queries
        
        Returns:
            List of recent queries
        """
        recent_turns = list(self.turns)[-n:]
        return [turn.query for turn in recent_turns]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get conversation summary.
        
        Returns:
            Summary dictionary
        """
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "age_hours": (datetime.utcnow() - self.created_at).total_seconds() / 3600,
            "is_expired": self.is_expired()
        }
    
    def is_expired(self) -> bool:
        """
        Check if session is expired.
        
        Returns:
            True if expired, False otherwise
        """
        age = datetime.utcnow() - self.last_accessed
        return age > timedelta(hours=self.ttl_hours)
    
    def clear(self):
        """Clear conversation history."""
        self.turns.clear()
        self._save_to_cache()
        logger.info(f"Cleared conversation history for session: {self.session_id}")
    
    def _get_cache_key(self) -> str:
        """Get Redis cache key for this session."""
        return f"conversation:session:{self.session_id}"
    
    def _save_to_cache(self):
        """Save conversation to Redis cache."""
        try:
            cache = get_cache()
            if not cache.enabled or not cache.redis_client:
                return
            
            data = {
                "session_id": self.session_id,
                "turns": [turn.to_dict() for turn in self.turns],
                "created_at": self.created_at.isoformat(),
                "last_accessed": self.last_accessed.isoformat(),
                "max_turns": self.max_turns,
                "context_window": self.context_window
            }
            
            key = self._get_cache_key()
            ttl = int(self.ttl_hours * 3600)  # Convert to seconds
            
            # Store as JSON
            cache.redis_client.setex(
                key,
                ttl,
                json.dumps(data)
            )
            
            logger.debug(f"Saved conversation to cache: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error saving conversation to cache: {e}")
    
    def _load_from_cache(self):
        """Load conversation from Redis cache."""
        try:
            cache = get_cache()
            if not cache.enabled or not cache.redis_client:
                return
            
            key = self._get_cache_key()
            cached_data = cache.redis_client.get(key)
            
            if cached_data:
                # Decode if bytes
                data_str = cached_data.decode('utf-8') if isinstance(cached_data, bytes) else str(cached_data)
                data = json.loads(data_str)
                
                # Restore turns
                self.turns = deque(
                    [ConversationTurn.from_dict(turn_data) for turn_data in data["turns"]],
                    maxlen=self.max_turns
                )
                
                self.created_at = datetime.fromisoformat(data["created_at"])
                self.last_accessed = datetime.fromisoformat(data["last_accessed"])
                
                logger.info(f"Loaded conversation from cache: {self.session_id} ({len(self.turns)} turns)")
            
        except Exception as e:
            logger.error(f"Error loading conversation from cache: {e}")


class ConversationManager:
    """
    Manages multiple conversation sessions.
    """
    
    def __init__(self):
        """Initialize conversation manager."""
        self.sessions: Dict[str, ConversationMemory] = {}
        logger.info("Initialized ConversationManager")
    
    def get_or_create_session(
        self,
        session_id: str,
        max_turns: int = 10,
        context_window: int = 3
    ) -> ConversationMemory:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum turns to store
            context_window: Context window size
        
        Returns:
            ConversationMemory instance
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationMemory(
                session_id=session_id,
                max_turns=max_turns,
                context_window=context_window
            )
        
        # Check if session is expired
        session = self.sessions[session_id]
        if session.is_expired():
            logger.info(f"Session expired, creating new: {session_id}")
            self.sessions[session_id] = ConversationMemory(
                session_id=session_id,
                max_turns=max_turns,
                context_window=context_window
            )
        
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.
        
        Returns:
            Number of sessions removed
        """
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired()
        ]
        
        for sid in expired:
            self.delete_session(sid)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
        
        return len(expired)
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of active sessions.
        
        Returns:
            List of session summaries
        """
        return [
            session.get_summary()
            for session in self.sessions.values()
            if not session.is_expired()
        ]


# Global conversation manager instance
_manager_instance: Optional[ConversationManager] = None


def get_conversation_manager() -> ConversationManager:
    """
    Get or create global conversation manager.
    
    Returns:
        ConversationManager instance
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ConversationManager()
    return _manager_instance


# Made with Bob