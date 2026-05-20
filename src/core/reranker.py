"""
Cross-encoder reranking implementation for improving retrieval quality.
Uses a cross-encoder model to rerank retrieved documents based on query relevance.
"""

from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
import numpy as np

from config.settings import settings
from src.utils.logger import get_logger
from src.models.schemas import RetrievedDocument

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder based reranker for improving retrieval results.
    Cross-encoders jointly encode query and document for better relevance scoring.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name or settings.CROSS_ENCODER_MODEL
        
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using cross-encoder model.
        
        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return (None = return all)
        
        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []
        
        try:
            # Prepare query-document pairs for cross-encoder
            pairs = [[query, doc.content] for doc in documents]
            
            # Get cross-encoder scores
            logger.debug(f"Reranking {len(pairs)} documents")
            scores = self.model.predict(pairs)
            
            # Normalize scores to [0, 1] range using sigmoid
            normalized_scores = self._sigmoid(scores)
            
            # Update documents with rerank scores
            for doc, score in zip(documents, normalized_scores):
                doc.rerank_score = float(score)
            
            # Sort by rerank score
            reranked_docs = sorted(
                documents,
                key=lambda x: x.rerank_score or 0.0,
                reverse=True
            )
            
            # Return top-k if specified
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]
            
            logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original documents if reranking fails
            return documents
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid function to normalize scores.
        
        Args:
            x: Input scores
        
        Returns:
            Normalized scores
        """
        return 1 / (1 + np.exp(-x))
    
    def get_relevance_score(self, query: str, document: str) -> float:
        """
        Get relevance score for a single query-document pair.
        
        Args:
            query: Search query
            document: Document text
        
        Returns:
            Relevance score between 0 and 1
        """
        try:
            score = self.model.predict([[query, document]])[0]
            return float(self._sigmoid(np.array([score]))[0])
        except Exception as e:
            logger.error(f"Error getting relevance score: {e}")
            return 0.0


class EnsembleReranker:
    """
    Ensemble reranker that combines multiple ranking signals.
    Combines vector similarity, BM25, and cross-encoder scores.
    """
    
    def __init__(
        self,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        weights: Tuple[float, float, float] = (0.3, 0.2, 0.5)
    ):
        """
        Initialize ensemble reranker.
        
        Args:
            cross_encoder: CrossEncoderReranker instance
            weights: Weights for (vector, bm25, cross_encoder) scores
        """
        self.cross_encoder = cross_encoder or CrossEncoderReranker()
        self.weights = weights
        
        # Validate weights sum to 1
        if not np.isclose(sum(weights), 1.0):
            logger.warning(f"Weights sum to {sum(weights)}, normalizing to 1.0")
            total = sum(weights)
            self.weights = tuple(w / total for w in weights)
        
        logger.info(f"Initialized EnsembleReranker with weights: {self.weights}")
    
    def rerank(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_k: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """
        Rerank documents using ensemble of multiple signals.
        
        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return
        
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        try:
            # First apply cross-encoder reranking
            documents = self.cross_encoder.rerank(query, documents)
            
            # Calculate ensemble scores
            for doc in documents:
                vector_score = doc.similarity_score or 0.0
                bm25_score = doc.bm25_score or 0.0
                rerank_score = doc.rerank_score or 0.0
                
                # Weighted combination
                ensemble_score = (
                    self.weights[0] * vector_score +
                    self.weights[1] * bm25_score +
                    self.weights[2] * rerank_score
                )
                
                # Store in hybrid_score field
                doc.hybrid_score = ensemble_score
            
            # Sort by ensemble score
            reranked_docs = sorted(
                documents,
                key=lambda x: x.hybrid_score or 0.0,
                reverse=True
            )
            
            # Return top-k if specified
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]
            
            logger.info(f"Ensemble reranking complete, returning {len(reranked_docs)} documents")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in ensemble reranking: {e}")
            return documents

# Made with Bob
