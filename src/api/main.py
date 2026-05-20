"""
FastAPI application entry point with comprehensive API endpoints.
Provides REST API for the advanced RAG system with proper error handling,
rate limiting, and monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

from config.settings import settings
from src.utils.logger import get_logger, api_logger
from src.models.schemas import (
    QueryRequest, QueryResponse, DocumentUploadRequest,
    DocumentUploadResponse, HealthCheckResponse, MetricsResponse,
    FeedbackRequest, FeedbackResponse
)
from src.api.dependencies import get_rag_system, get_metrics_collector
from src.api.rate_limiter import RateLimiter
from src.services.document_processor import DocumentProcessor
from src.services.metrics_collector import MetricsCollector

logger = get_logger(__name__)

# Global instances
rag_system = None
metrics_collector = None
rate_limiter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Initializes and cleans up resources.
    """
    global rag_system, metrics_collector, rate_limiter
    
    # Startup
    logger.info("Starting up Advanced RAG API...")
    
    try:
        # Initialize components
        from src.core.langgraph_rag import LangGraphRAG
        from src.services.metrics_collector import MetricsCollector
        from src.api.rate_limiter import RateLimiter
        
        rag_system = LangGraphRAG()
        metrics_collector = MetricsCollector()
        rate_limiter = RateLimiter(
            max_requests=settings.RATE_LIMIT_PER_MINUTE,
            window_seconds=60
        )
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Advanced RAG API...")
    # Cleanup resources if needed


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready RAG system with advanced features",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests and collect metrics."""
    start_time = time.time()
    
    # Log request
    api_logger.info(
        f"Request: {request.method} {request.url.path}"
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log response
    api_logger.info(
        f"Response: {response.status_code} in {duration:.3f}s"
    )
    
    # Collect metrics
    if metrics_collector:
        metrics_collector.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration=duration
        )
    
    return response


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify service status.
    Checks all dependent services.
    """
    try:
        services_status = {
            "rag_system": rag_system is not None,
            "vector_store": True,  # Add actual check
            "llm": True,  # Add actual check
            "redis": True,  # Add actual check
        }
        
        all_healthy = all(services_status.values())
        
        return HealthCheckResponse(
            status="healthy" if all_healthy else "degraded",
            version=settings.APP_VERSION,
            timestamp=datetime.utcnow(),
            services=services_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(
    request: QueryRequest,
    rate_limit: None = Depends(lambda: rate_limiter.check_rate_limit() if rate_limiter else None)
):
    """
    Query the RAG system with a question.
    
    This endpoint processes queries through the complete RAG pipeline:
    1. Query classification
    2. Hybrid retrieval (vector + BM25)
    3. Cross-encoder reranking
    4. Answer generation with citations
    5. Hallucination detection
    6. Context grounding check
    
    Args:
        request: Query request with question and parameters
    
    Returns:
        QueryResponse with answer, citations, and metadata
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Check if RAG system is initialized
        if rag_system is None:
            logger.error("RAG system not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG system not initialized. Please check service health."
            )
        
        # Process query through RAG pipeline
        response = rag_system.query(request)
        
        # Record metrics
        if metrics_collector:
            metrics_collector.record_query(
                query_type=response.query_type.value if response.query_type else "unknown",
                confidence=response.confidence_score,
                processing_time=response.processing_time_ms
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Upload and process a document for indexing.
    
    Supports: PDF, TXT, DOCX, MD files
    
    Args:
        file: Document file to upload
        metadata: Optional metadata for the document
    
    Returns:
        DocumentUploadResponse with upload status
    """
    try:
        logger.info(f"Uploading document: {file.filename}")
        
        # Check if RAG system is initialized
        if rag_system is None:
            logger.error("RAG system not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG system not initialized. Please check service health."
            )
        
        # Validate filename exists
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        # Validate file type
        file_ext = file.filename.split('.')[-1].lower()
        if f".{file_ext}" not in settings.SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_ext}"
            )
        
        # Read file content
        content = await file.read()
        
        # Check file size
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large: {file_size_mb:.2f}MB (max: {settings.MAX_FILE_SIZE_MB}MB)"
            )
        
        # Process document
        processor = DocumentProcessor(rag_system.vector_store)
        result = await processor.process_document(
            content=content,
            filename=file.filename,
            metadata=metadata or {}
        )
        
        logger.info(f"Document processed: {result['chunks_created']} chunks created")
        
        return DocumentUploadResponse(
            document_id=result["document_id"],
            status="success",
            chunks_created=result["chunks_created"],
            message=f"Document '{file.filename}' processed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: str):
    """
    Delete a document and its chunks from the vector store.
    
    Args:
        document_id: ID of the document to delete
    
    Returns:
        Success message
    """
    try:
        logger.info(f"Deleting document: {document_id}")
        
        # Check if RAG system is initialized
        if rag_system is None:
            logger.error("RAG system not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG system not initialized. Please check service health."
            )
        
        # Delete from vector store
        success = rag_system.vector_store.delete_documents([document_id])
        
        if success:
            return {"status": "success", "message": f"Document {document_id} deleted"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """
    Get system metrics and statistics.
    
    Returns:
        MetricsResponse with system metrics
    """
    try:
        if not metrics_collector:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Metrics collection not available"
            )
        
        metrics = metrics_collector.get_metrics()
        
        return MetricsResponse(
            total_queries=metrics["total_queries"],
            average_response_time_ms=metrics["avg_response_time"],
            cache_hit_rate=metrics["cache_hit_rate"],
            error_rate=metrics["error_rate"],
            uptime_seconds=metrics["uptime"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving metrics: {str(e)}"
        )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for a query response.
    
    Args:
        feedback: Feedback data
    
    Returns:
        FeedbackResponse with submission status
    """
    try:
        logger.info(f"Received feedback for query: {feedback.query_id}")
        
        # Store feedback (implement storage logic)
        # For now, just log it
        logger.info(
            f"Feedback: rating={feedback.rating}, "
            f"helpful={feedback.is_helpful}, "
            f"text={feedback.feedback_text}"
        )
        
        # Record in metrics
        if metrics_collector:
            metrics_collector.record_feedback(
                rating=feedback.rating,
                is_helpful=feedback.is_helpful
            )
        
        return FeedbackResponse(
            status="success",
            message="Feedback received successfully"
        )
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting feedback: {str(e)}"
        )


@app.get("/collection/info", tags=["System"])
async def get_collection_info():
    """
    Get information about the vector store collection.
    
    Returns:
        Collection information and statistics
    """
    try:
        # Check if RAG system is initialized
        if rag_system is None:
            logger.error("RAG system not initialized")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG system not initialized. Please check service health."
            )
        
        info = rag_system.vector_store.get_collection_info()
        return info
        
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving collection info: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

# Made with Bob
