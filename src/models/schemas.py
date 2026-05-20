"""
Pydantic models for request/response validation and data structures.
Ensures type safety and automatic validation across the application.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"


class DocumentMetadata(BaseModel):
    """Metadata associated with a document chunk."""
    source: str = Field(..., description="Source file path or URL")
    page: Optional[int] = Field(None, description="Page number if applicable")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    file_type: str = Field(..., description="Type of source file")
    section: Optional[str] = Field(None, description="Section or heading")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Citation(BaseModel):
    """Citation information for a source."""
    source: str = Field(..., description="Source identifier")
    page: Optional[int] = Field(None, description="Page number")
    chunk_id: str = Field(..., description="Chunk identifier")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    text_snippet: str = Field(..., description="Relevant text excerpt")


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    query_type: Optional[QueryType] = Field(None, description="Type of query")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of results to retrieve")
    include_sources: bool = Field(True, description="Include source citations")
    session_id: Optional[str] = Field(None, description="Session identifier for context")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty after stripping."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class RetrievedDocument(BaseModel):
    """A retrieved document with metadata and scores."""
    content: str = Field(..., description="Document content")
    metadata: DocumentMetadata = Field(..., description="Document metadata")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Vector similarity score")
    rerank_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Cross-encoder rerank score")
    bm25_score: Optional[float] = Field(None, description="BM25 keyword score")
    hybrid_score: Optional[float] = Field(None, description="Combined hybrid score")


class HallucinationCheck(BaseModel):
    """Result of hallucination detection."""
    is_hallucinated: bool = Field(..., description="Whether response contains hallucinations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in detection")
    explanation: Optional[str] = Field(None, description="Explanation of detection")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims not supported by context")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list, description="Retrieved documents")
    hallucination_check: Optional[HallucinationCheck] = Field(None, description="Hallucination detection result")
    query_type: Optional[QueryType] = Field(None, description="Detected query type")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in answer")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    file_name: str = Field(..., description="Name of the file")
    file_type: str = Field(..., description="Type of file (pdf, txt, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Unique document identifier")
    status: str = Field(..., description="Upload status")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Status message")


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for RAG system."""
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Faithfulness score")
    answer_relevancy: float = Field(..., ge=0.0, le=1.0, description="Answer relevancy score")
    context_precision: float = Field(..., ge=0.0, le=1.0, description="Context precision score")
    context_recall: float = Field(..., ge=0.0, le=1.0, description="Context recall score")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, bool] = Field(..., description="Status of dependent services")


class MetricsResponse(BaseModel):
    """System metrics response."""
    total_queries: int = Field(..., description="Total queries processed")
    average_response_time_ms: float = Field(..., description="Average response time")
    cache_hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit rate")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate")
    uptime_seconds: float = Field(..., description="System uptime in seconds")


class FeedbackRequest(BaseModel):
    """User feedback on a response."""
    query_id: str = Field(..., description="Query identifier")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(None, description="Optional feedback text")
    is_helpful: bool = Field(..., description="Whether response was helpful")


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    status: str = Field(..., description="Feedback submission status")
    message: str = Field(..., description="Status message")

# Made with Bob
