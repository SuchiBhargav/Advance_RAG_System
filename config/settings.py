"""
Production-ready configuration management with environment-based settings.
Supports multiple environments: development, staging, production.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """
    Application settings with validation and environment variable support.
    All sensitive data should be loaded from environment variables.
    """
    
    # Application Settings
    APP_NAME: str = "Advanced RAG System"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=True)
    
    # API Settings
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_WORKERS: int = Field(default=4)
    CORS_ORIGINS: List[str] = Field(default=["*"])
    
    # LLM Settings
    LLM_MODEL: str = Field(default="llama3")
    LLM_TEMPERATURE: float = Field(default=0.1)
    LLM_MAX_TOKENS: int = Field(default=2048)
    EMBEDDING_MODEL: str = Field(default="llama3")
    
    # Vector Database Settings (Qdrant)
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_COLLECTION_NAME: str = Field(default="documents")
    QDRANT_API_KEY: Optional[str] = Field(default=None)
    VECTOR_DIMENSION: int = Field(default=4096)
    
    # Redis Cache Settings
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    CACHE_TTL: int = Field(default=86400)  # 24 hours
    
    # Retrieval Settings
    RETRIEVAL_TOP_K: int = Field(default=10)
    RERANK_TOP_K: int = Field(default=5)
    SIMILARITY_THRESHOLD: float = Field(default=0.7)
    HYBRID_SEARCH_ALPHA: float = Field(default=0.5)  # 0=BM25, 1=vector
    
    # Cross-Encoder Settings
    CROSS_ENCODER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    
    # Document Processing Settings
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)
    MAX_FILE_SIZE_MB: int = Field(default=50)
    SUPPORTED_FILE_TYPES: List[str] = Field(
        default=[".pdf", ".txt", ".docx", ".md"]
    )
    
    # Paths
    DATA_DIR: str = Field(default="data")
    RAW_DATA_DIR: str = Field(default="data/raw")
    PROCESSED_DATA_DIR: str = Field(default="data/processed")
    VECTOR_STORE_DIR: str = Field(default="data/vector_store")
    LOG_DIR: str = Field(default="logs")
    
    # Logging Settings
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: str = Field(default="logs/app.log")
    
    # LangSmith/LangChain Settings
    LANGCHAIN_TRACING_V2: bool = Field(default=True)
    LANGCHAIN_API_KEY: Optional[str] = Field(default=None)
    LANGCHAIN_PROJECT: str = Field(default="advanced-rag")
    
    # Evaluation Settings
    ENABLE_EVALUATION: bool = Field(default=True)
    EVALUATION_SAMPLE_SIZE: int = Field(default=100)
    
    # Hallucination Detection Settings
    HALLUCINATION_THRESHOLD: float = Field(default=0.8)
    ENABLE_HALLUCINATION_CHECK: bool = Field(default=True)
    
    # Query Rewriting Settings
    ENABLE_QUERY_REWRITE: bool = Field(default=True)
    QUERY_REWRITE_THRESHOLD: float = Field(default=0.6)
    MAX_REWRITE_ATTEMPTS: int = Field(default=2)
    
    # Conversation Memory Settings
    ENABLE_CONVERSATION_MEMORY: bool = Field(default=True)
    MAX_CONVERSATION_TURNS: int = Field(default=10)
    CONVERSATION_CONTEXT_WINDOW: int = Field(default=3)
    CONVERSATION_TTL_HOURS: int = Field(default=24)
    
    # Security Settings
    ENABLE_PROMPT_INJECTION_DETECTION: bool = Field(default=True)
    BLOCK_HIGH_RISK_QUERIES: bool = Field(default=True)
    
    # Retry Logic Settings
    ENABLE_RETRY_ON_LOW_CONFIDENCE: bool = Field(default=True)
    RETRY_CONFIDENCE_THRESHOLD: float = Field(default=0.5)
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60)
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=9090)
    PROMETHEUS_ENABLED: bool = Field(default=True)
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment is one of the allowed values."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings instance.
    
    Returns:
        Settings: Application settings
    """
    return settings

# Made with Bob
