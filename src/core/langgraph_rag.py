"""
LangGraph-based RAG orchestration with state management and advanced workflow.
Implements a sophisticated RAG pipeline with query routing, retrieval, reranking,
generation, hallucination detection, and citation extraction.
"""

from typing import TypedDict, List, Dict, Any, Annotated, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time

from config.settings import settings
from src.utils.logger import get_logger
from src.models.schemas import (
    QueryRequest, QueryResponse, RetrievedDocument,
    Citation, QueryType, HallucinationCheck
)
from src.core.vector_store import HybridVectorStore
from src.core.reranker import EnsembleReranker
from src.core.hallucination_detector import HallucinationDetector, ContextGroundingChecker

logger = get_logger(__name__)


class RAGState(TypedDict):
    """
    State object for the RAG workflow.
    Tracks all information through the pipeline.
    """
    # Input
    query: str
    query_type: QueryType
    filters: Dict[str, Any]
    top_k: int
    
    # Retrieval
    retrieved_documents: List[RetrievedDocument]
    reranked_documents: List[RetrievedDocument]
    
    # Generation
    context: str
    answer: str
    citations: List[Citation]
    
    # Quality checks
    hallucination_check: Optional[HallucinationCheck]
    grounding_metrics: Dict[str, Any]
    
    # Metadata
    confidence_score: float
    processing_steps: List[str]
    start_time: float
    
    # Error handling
    error: str


class LangGraphRAG:
    """
    Advanced RAG system using LangGraph for orchestration.
    Implements a multi-stage pipeline with quality controls.
    """
    
    def __init__(
        self,
        vector_store: Optional[HybridVectorStore] = None,
        reranker: Optional[EnsembleReranker] = None,
        hallucination_detector: Optional[HallucinationDetector] = None
    ):
        """
        Initialize LangGraph RAG system.
        
        Args:
            vector_store: Vector store instance
            reranker: Reranker instance
            hallucination_detector: Hallucination detector instance
        """
        self.vector_store = vector_store or HybridVectorStore()
        self.reranker = reranker or EnsembleReranker()
        self.hallucination_detector = hallucination_detector or HallucinationDetector()
        self.grounding_checker = ContextGroundingChecker()
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("Initialized LangGraphRAG system")
    
    def _build_workflow(self) -> CompiledStateGraph:
        """
        Build the LangGraph workflow for RAG pipeline.
        
        Returns:
            Compiled StateGraph
        """
        # Create workflow graph
        workflow = StateGraph(RAGState)
        
        # Add nodes for each stage
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("rerank", self._rerank_documents)
        workflow.add_node("generate", self._generate_answer)
        workflow.add_node("extract_citations", self._extract_citations)
        workflow.add_node("check_hallucination", self._check_hallucination)
        workflow.add_node("check_grounding", self._check_grounding)
        workflow.add_node("finalize", self._finalize_response)
        
        # Define workflow edges
        workflow.set_entry_point("classify_query")
        workflow.add_edge("classify_query", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "extract_citations")
        workflow.add_edge("extract_citations", "check_hallucination")
        workflow.add_edge("check_hallucination", "check_grounding")
        workflow.add_edge("check_grounding", "finalize")
        workflow.add_edge("finalize", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _classify_query(self, state: RAGState) -> RAGState:
        """
        Classify the query type for better routing.
        
        Args:
            state: Current RAG state
        
        Returns:
            Updated state with query classification
        """
        logger.info("Step 1: Classifying query")
        state["processing_steps"].append("classify_query")
        
        query = state["query"].lower()
        
        # Simple rule-based classification
        if any(word in query for word in ["how", "why", "explain", "describe"]):
            state["query_type"] = QueryType.ANALYTICAL
        elif any(word in query for word in ["what", "when", "where", "who"]):
            state["query_type"] = QueryType.FACTUAL
        elif any(word in query for word in ["error", "issue", "problem", "fix", "debug"]):
            state["query_type"] = QueryType.TECHNICAL
        else:
            state["query_type"] = QueryType.CONVERSATIONAL
        
        logger.info(f"Query classified as: {state['query_type']}")
        return state
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents using hybrid search.
        
        Args:
            state: Current RAG state
        
        Returns:
            Updated state with retrieved documents
        """
        logger.info("Step 2: Retrieving documents")
        state["processing_steps"].append("retrieve")
        
        try:
            # Perform hybrid search
            documents = self.vector_store.hybrid_search(
                query=state["query"],
                top_k=state["top_k"] * 2,  # Retrieve more for reranking
                alpha=settings.HYBRID_SEARCH_ALPHA,
                filters=state.get("filters")
            )
            
            state["retrieved_documents"] = documents
            logger.info(f"Retrieved {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            state["error"] = f"Retrieval error: {str(e)}"
            state["retrieved_documents"] = []
        
        return state
    
    def _rerank_documents(self, state: RAGState) -> RAGState:
        """
        Rerank retrieved documents using cross-encoder.
        
        Args:
            state: Current RAG state
        
        Returns:
            Updated state with reranked documents
        """
        logger.info("Step 3: Reranking documents")
        state["processing_steps"].append("rerank")
        
        try:
            if not state["retrieved_documents"]:
                logger.warning("No documents to rerank")
                state["reranked_documents"] = []
                return state
            
            # Rerank using ensemble reranker
            reranked = self.reranker.rerank(
                query=state["query"],
                documents=state["retrieved_documents"],
                top_k=state["top_k"]
            )
            
            state["reranked_documents"] = reranked
            logger.info(f"Reranked to top {len(reranked)} documents")
            
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            state["error"] = f"Reranking error: {str(e)}"
            # Fall back to retrieved documents
            state["reranked_documents"] = state["retrieved_documents"][:state["top_k"]]
        
        return state
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            state: Current RAG state
        
        Returns:
            Updated state with generated answer
        """
        logger.info("Step 4: Generating answer")
        state["processing_steps"].append("generate")
        
        try:
            if not state["reranked_documents"]:
                state["answer"] = "I don't have enough information to answer this question."
                state["context"] = ""
                state["confidence_score"] = 0.0
                return state
            
            # Prepare context from reranked documents
            context_parts = []
            for i, doc in enumerate(state["reranked_documents"], 1):
                context_parts.append(
                    f"[Source {i}] (from {doc.metadata.source}, "
                    f"relevance: {doc.rerank_score or doc.similarity_score:.2f})\n"
                    f"{doc.content}"
                )
            
            context = "\n\n".join(context_parts)
            state["context"] = context
            
            # Create prompt with citations requirement
            prompt = ChatPromptTemplate.from_template("""
You are a technical assistant providing accurate, well-sourced answers.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY using the provided context
2. Include citations using [Source N] format for every claim
3. If information is not in the context, say "I don't know"
4. Be precise and factual
5. Do not make assumptions or add information not in the sources

Context:
{context}

Question: {query}

Answer (with citations):""")
            
            # Generate answer
            chain = prompt | self.llm
            response = chain.invoke({
                "context": context,
                "query": state["query"]
            })
            
            state["answer"] = response
            
            # Calculate confidence based on document scores
            avg_score = sum(
                doc.rerank_score or doc.similarity_score or 0.0
                for doc in state["reranked_documents"]
            ) / len(state["reranked_documents"])
            state["confidence_score"] = avg_score
            
            logger.info(f"Generated answer with confidence: {avg_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["error"] = f"Generation error: {str(e)}"
            state["answer"] = "An error occurred while generating the answer."
            state["confidence_score"] = 0.0
        
        return state
    
    def _extract_citations(self, state: RAGState) -> RAGState:
        """
        Extract and validate citations from the generated answer.
        
        Args:
            state: Current RAG state
        
        Returns:
            Updated state with extracted citations
        """
        logger.info("Step 5: Extracting citations")
        state["processing_steps"].append("extract_citations")
        
        try:
            citations = []
            answer = state["answer"]
            
            # Find all [Source N] references in answer
            import re
            citation_pattern = r'\[Source (\d+)\]'
            cited_sources = set(re.findall(citation_pattern, answer))
            
            # Create citation objects for cited sources
            for source_num in cited_sources:
                idx = int(source_num) - 1
                if idx < len(state["reranked_documents"]):
                    doc = state["reranked_documents"][idx]
                    
                    # Extract relevant snippet (first 200 chars)
                    snippet = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                    
                    citation = Citation(
                        source=doc.metadata.source,
                        page=doc.metadata.page,
                        chunk_id=doc.metadata.chunk_id,
                        relevance_score=doc.rerank_score or doc.similarity_score or 0.0,
                        text_snippet=snippet
                    )
                    citations.append(citation)
            
            state["citations"] = citations
            logger.info(f"Extracted {len(citations)} citations")
            
        except Exception as e:
            logger.error(f"Error extracting citations: {e}")
            state["citations"] = []
        
        return state
    
    def _check_hallucination(self, state: RAGState) -> RAGState:
        """
        Check for hallucinations in the generated answer.
        
        Args:
            state: Current RAG state
        
        Returns:
            Updated state with hallucination check results
        """
        logger.info("Step 6: Checking for hallucinations")
        state["processing_steps"].append("check_hallucination")
        
        try:
            if settings.ENABLE_HALLUCINATION_CHECK:
                hallucination_check = self.hallucination_detector.detect(
                    response=state["answer"],
                    context_documents=state["reranked_documents"]
                )
                state["hallucination_check"] = hallucination_check
                
                # Adjust confidence if hallucination detected
                if hallucination_check.is_hallucinated:
                    state["confidence_score"] *= 0.5  # Reduce confidence
                    logger.warning("Hallucination detected, confidence reduced")
            else:
                state["hallucination_check"] = None
                
        except Exception as e:
            logger.error(f"Error checking hallucination: {e}")
            state["hallucination_check"] = None
        
        return state
    
    def _check_grounding(self, state: RAGState) -> RAGState:
        """
        Check if answer is well-grounded in context.
        
        Args:
            state: Current RAG state
        
        Returns:
            Updated state with grounding metrics
        """
        logger.info("Step 7: Checking context grounding")
        state["processing_steps"].append("check_grounding")
        
        try:
            grounding_metrics = self.grounding_checker.check_grounding(
                response=state["answer"],
                context_documents=state["reranked_documents"]
            )
            state["grounding_metrics"] = grounding_metrics
            
            # Adjust confidence based on grounding
            if not grounding_metrics.get("is_well_grounded", True):
                state["confidence_score"] *= 0.7
                logger.warning("Poor grounding detected, confidence reduced")
                
        except Exception as e:
            logger.error(f"Error checking grounding: {e}")
            state["grounding_metrics"] = {}
        
        return state
    
    def _finalize_response(self, state: RAGState) -> RAGState:
        """
        Finalize the response with metadata.
        
        Args:
            state: Current RAG state
        
        Returns:
            Final state
        """
        logger.info("Step 8: Finalizing response")
        state["processing_steps"].append("finalize")
        
        # Calculate processing time
        processing_time = (time.time() - state["start_time"]) * 1000  # Convert to ms
        
        logger.info(
            f"RAG pipeline complete in {processing_time:.2f}ms, "
            f"confidence: {state['confidence_score']:.2f}"
        )
        
        return state
    
    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            request: Query request object
        
        Returns:
            Query response with answer and metadata
        """
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Initialize state
        initial_state: RAGState = {
            "query": request.query,
            "query_type": request.query_type or QueryType.FACTUAL,
            "filters": request.filters or {},
            "top_k": request.top_k or settings.RERANK_TOP_K,
            "retrieved_documents": [],
            "reranked_documents": [],
            "context": "",
            "answer": "",
            "citations": [],
            "hallucination_check": None,
            "grounding_metrics": {},
            "confidence_score": 0.0,
            "processing_steps": [],
            "start_time": time.time(),
            "error": ""
        }
        
        try:
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Calculate processing time
            processing_time = (time.time() - final_state["start_time"]) * 1000
            
            # Build response
            response = QueryResponse(
                answer=final_state["answer"],
                citations=final_state["citations"] if request.include_sources else [],
                retrieved_documents=final_state["reranked_documents"] if request.include_sources else [],
                hallucination_check=final_state.get("hallucination_check"),
                query_type=final_state["query_type"],
                confidence_score=final_state["confidence_score"],
                processing_time_ms=processing_time,
                metadata={
                    "processing_steps": final_state["processing_steps"],
                    "grounding_metrics": final_state.get("grounding_metrics", {}),
                    "error": final_state.get("error", "")
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            
            # Return error response
            return QueryResponse(
                answer="An error occurred while processing your query.",
                citations=[],
                retrieved_documents=[],
                hallucination_check=None,
                query_type=QueryType.FACTUAL,
                confidence_score=0.0,
                processing_time_ms=0.0,
                metadata={"error": str(e)}
            )

# Made with Bob
