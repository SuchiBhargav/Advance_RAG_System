"""
Advanced vector store implementation using Qdrant with hybrid search capabilities.
Supports vector similarity search, BM25 keyword search, and hybrid retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, SearchRequest,
    ScoredPoint, UpdateStatus, PointIdsList
)
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np

from config.settings import settings
from src.utils.logger import get_logger
from src.models.schemas import DocumentMetadata, RetrievedDocument

logger = get_logger(__name__)


class HybridVectorStore:
    """
    Advanced vector store with hybrid search capabilities.
    Combines dense vector search (semantic) with sparse BM25 search (keyword).
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        embeddings: Optional[OllamaEmbeddings] = None
    ):
        """
        Initialize the hybrid vector store.
        
        Args:
            collection_name: Name of the Qdrant collection
            embeddings: Embedding model instance
        """
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.embeddings = embeddings or OllamaEmbeddings(model=settings.EMBEDDING_MODEL)
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False,
            https=False
        )
        
        # BM25 index for keyword search (in-memory)
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_documents: List[Dict[str, Any]] = []
        
        # Initialize collection
        self._initialize_collection()
        
        logger.info(f"Initialized HybridVectorStore with collection: {self.collection_name}")
    
    def _initialize_collection(self) -> None:
        """
        Initialize or verify Qdrant collection exists.
        Creates collection with proper vector configuration if it doesn't exist.
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                logger.info(f"Creating new collection: {self.collection_name}")
                
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.VECTOR_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store with embeddings and metadata.
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dictionaries
            ids: Optional list of document IDs
        
        Returns:
            List of document IDs
        """
        if not texts:
            logger.warning("No texts provided to add_documents")
            return []
        
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Generate embeddings for all texts
            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = self.embeddings.embed_documents(texts)
            
            # Prepare points for Qdrant
            points = []
            for idx, (text, embedding, metadata, doc_id) in enumerate(
                zip(texts, embeddings, metadatas, ids)
            ):
                # Add timestamp if not present
                if "created_at" not in metadata:
                    metadata["created_at"] = datetime.utcnow().isoformat()
                
                # Store full text in payload
                payload = {
                    "text": text,
                    "metadata": metadata,
                    "chunk_id": doc_id
                }
                
                point = PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            # Upload to Qdrant
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            if operation_info.status == UpdateStatus.COMPLETED:
                logger.info(f"Successfully added {len(points)} documents to Qdrant")
            
            # Update BM25 index
            self._update_bm25_index(texts, ids, metadatas)
            
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def _update_bm25_index(
        self,
        texts: List[str],
        ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Update the BM25 index with new documents.
        
        Args:
            texts: Document texts
            ids: Document IDs
            metadatas: Document metadata
        """
        try:
            # Add to BM25 documents
            for text, doc_id, metadata in zip(texts, ids, metadatas):
                self.bm25_documents.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "tokens": text.lower().split()
                })
            
            # Rebuild BM25 index
            tokenized_corpus = [doc["tokens"] for doc in self.bm25_documents]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            
            logger.info(f"Updated BM25 index with {len(self.bm25_documents)} documents")
            
        except Exception as e:
            logger.error(f"Error updating BM25 index: {e}")
    
    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters
        
        Returns:
            List of (doc_id, score, payload) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Build filter if provided
            qdrant_filter = None
            if filters:
                qdrant_filter = self._build_filter(filters)
            
            # Search in Qdrant
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter
            ).points
            
            # Extract results
            results = [
                (str(point.id), point.score, point.payload or {})
                for point in search_result
            ]
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def bm25_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (doc_id, score, document) tuples
        """
        if not self.bm25_index or not self.bm25_documents:
            logger.warning("BM25 index not initialized")
            return []
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            # Build results
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include non-zero scores
                    doc = self.bm25_documents[idx]
                    results.append((
                        doc["id"],
                        float(scores[idx]),
                        {"text": doc["text"], "metadata": doc["metadata"]}
                    ))
            
            logger.debug(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Perform hybrid search combining vector and BM25 search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for vector search (0=BM25 only, 1=vector only)
            filters: Metadata filters
        
        Returns:
            List of RetrievedDocument objects
        """
        try:
            # Perform both searches
            vector_results = self.vector_search(query, top_k * 2, filters)
            bm25_results = self.bm25_search(query, top_k * 2)
            
            # Normalize scores
            vector_scores = self._normalize_scores([r[1] for r in vector_results])
            bm25_scores = self._normalize_scores([r[1] for r in bm25_results])
            
            # Combine results with weighted scores
            combined_scores: Dict[str, Dict[str, Any]] = {}
            
            # Add vector results
            for (doc_id, _, payload), norm_score in zip(vector_results, vector_scores):
                combined_scores[doc_id] = {
                    "payload": payload,
                    "vector_score": norm_score,
                    "bm25_score": 0.0,
                    "hybrid_score": alpha * norm_score
                }
            
            # Add BM25 results
            for (doc_id, _, payload), norm_score in zip(bm25_results, bm25_scores):
                if doc_id in combined_scores:
                    combined_scores[doc_id]["bm25_score"] = norm_score
                    combined_scores[doc_id]["hybrid_score"] += (1 - alpha) * norm_score
                else:
                    combined_scores[doc_id] = {
                        "payload": payload,
                        "vector_score": 0.0,
                        "bm25_score": norm_score,
                        "hybrid_score": (1 - alpha) * norm_score
                    }
            
            # Sort by hybrid score and take top-k
            sorted_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1]["hybrid_score"],
                reverse=True
            )[:top_k]
            
            # Convert to RetrievedDocument objects
            retrieved_docs = []
            for doc_id, scores in sorted_results:
                payload = scores["payload"]
                
                # Create metadata object
                # Safely extract metadata with proper type handling
                meta_dict = payload.get("metadata", {})
                metadata = DocumentMetadata(
                    source=meta_dict.get("source", "unknown"),
                    page=meta_dict.get("page"),
                    chunk_id=doc_id,
                    file_type=meta_dict.get("file_type", "unknown"),
                    section=meta_dict.get("section"),
                    metadata=meta_dict
                )
                
                retrieved_doc = RetrievedDocument(
                    content=payload.get("text", ""),
                    metadata=metadata,
                    similarity_score=scores["vector_score"],
                    bm25_score=scores["bm25_score"],
                    hybrid_score=scores["hybrid_score"],
                    rerank_score=None
                )
                retrieved_docs.append(retrieved_doc)
            
            logger.info(f"Hybrid search returned {len(retrieved_docs)} results")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: List of scores
        
        Returns:
            Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from metadata filters.
        
        Args:
            filters: Dictionary of filter conditions
        
        Returns:
            Qdrant Filter object
        """
        conditions = []
        
        for key, value in filters.items():
            condition = FieldCondition(
                key=f"metadata.{key}",
                match=MatchValue(value=value)
            )
            conditions.append(condition)
        
        return Filter(must=conditions)
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        
        Returns:
            True if successful
        """
        try:
            # Use PointIdsList with proper type casting
            # Type ignore is needed as qdrant-client accepts strings but type hints are strict
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids)  # type: ignore
            )
            
            # Remove from BM25 index
            self.bm25_documents = [
                doc for doc in self.bm25_documents if doc["id"] not in ids
            ]
            
            # Rebuild BM25 index
            if self.bm25_documents:
                tokenized_corpus = [doc["tokens"] for doc in self.bm25_documents]
                self.bm25_index = BM25Okapi(tokenized_corpus)
            
            logger.info(f"Deleted {len(ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": getattr(collection_info, 'vectors_count', 0),
                "points_count": getattr(collection_info, 'points_count', 0),
                "status": str(getattr(collection_info, 'status', 'unknown'))
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}

# Made with Bob
