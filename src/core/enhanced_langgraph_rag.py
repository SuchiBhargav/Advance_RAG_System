"""
Enhanced LangGraph RAG with query rewriting, retry logic, caching, and conversation memory.
Production-ready implementation with all advanced features.
"""

from typing import TypedDict, List, Dict, Any, Annotated, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time
import uuid

from config.settings import settings
from src.utils.logger import get_logger
from src.models.schemas import (
    QueryRequest, QueryResponse, RetrievedDocument,
    Citation, QueryType, HallucinationCheck
)
from src.core.vector_store import HybridVectorStore
from src.core.reranker import EnsembleReranker
from src.core.hallucination_detector import HallucinationDetector, ContextGroundingChecker
from src.core.query_rewriter import QueryRewriter, QueryClassifier, PromptInjectionDetector
from src.core.conversation_memory import get_conversation_manager
from src.services.redis_cache import get_cache

logger = get_logger(__name__)


class EnhancedRAGState(TypedDict):
    """Enhanced state object for the RAG workflow with retry and rewrite support."""
    # Input
    query: str
    original_query: str
    session_id: Optional[str]
    query_type: QueryType
    filters: Dict[str, Any]
    top_k: int
    
    # Query processing
    query_classification: Dict[str, Any]
    injection_check: Dict[str, Any]
    rewrite_attempts: int
    query_rewritten: bool
    
    # Retrieval
    retrieved_documents: List[RetrievedDocument]
    reranked_documents: List[RetrievedDocument]
    
    # Generation
    context: str
    answer: str
    citations: List[Citation]
    conversation_context: str
    
    # Quality checks
    hallucination_check: Optional[HallucinationCheck]
    grounding_metrics: Dict[str, Any]
    
    # Metadata
    confidence_score: float
    processing_steps: List[str]
    start_time: float
    should_retry: bool
    retry_count: int
    
    # Error handling
    error: str
    blocked: bool
    block_reason: str


class EnhancedLangGraphRAG:
    """
    Production-ready RAG system with all advanced features:
    - Redis caching
    - Query rewriting with retry logic
    - Prompt injection detection
    - Conversation memory
    - Conditional branching based on confidence
    """
    
    def __init__(
        self,
        vector_store: Optional[HybridVectorStore] = None,
        reranker: Optional[EnsembleReranker] = None,
        hallucination_detector: Optional[HallucinationDetector] = None
    ):
        """Initialize enhanced RAG system with all components."""
        self.vector_store = vector_store or HybridVectorStore()
        self.reranker = reranker or EnsembleReranker()
        self.hallucination_detector = hallucination_detector or HallucinationDetector()
        self.grounding_checker = ContextGroundingChecker()
        self.query_rewriter = QueryRewriter()
        self.query_classifier = QueryClassifier()
        self.injection_detector = PromptInjectionDetector()
        self.conversation_manager = get_conversation_manager()
        self.cache = get_cache()
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("Initialized EnhancedLangGraphRAG system with all features")
    
    def _build_workflow(self) -> CompiledStateGraph:
        """Build enhanced workflow with conditional branching and retry logic."""
        workflow = StateGraph(EnhancedRAGState)
        
        # Add nodes
        workflow.add_node("check_cache", self._check_cache)
        workflow.add_node("security_check", self._security_check)
        workflow.add_node("load_conversation", self._load_conversation)
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("rerank", self._rerank_documents)
        workflow.add_node("generate", self._generate_answer)
        workflow.add_node("extract_citations", self._extract_citations)
        workflow.add_node("check_hallucination", self._check_hallucination)
        workflow.add_node("check_grounding", self._check_grounding)
        workflow.add_node("evaluate_confidence", self._evaluate_confidence)
        workflow.add_node("save_conversation", self._save_conversation)
        workflow.add_node("cache_response", self._cache_response)
        workflow.add_node("finalize", self._finalize_response)
        workflow.add_node("handle_blocked", self._handle_blocked)
        
        # Define workflow with conditional edges
        workflow.set_entry_point("check_cache")
        
        # Cache check -> security or finalize
        workflow.add_conditional_edges(
            "check_cache",
            self._route_after_cache,
            {
                "cached": "finalize",
                "not_cached": "security_check"
            }
        )
        
        # Security check -> blocked or continue
        workflow.add_conditional_edges(
            "security_check",
            self._route_after_security,
            {
                "blocked": "handle_blocked",
                "safe": "load_conversation"
            }
        )
        
        workflow.add_edge("handle_blocked", "finalize")
        workflow.add_edge("load_conversation", "classify_query")
        
        # Classify -> rewrite or retrieve
        workflow.add_conditional_edges(
            "classify_query",
            self._route_after_classification,
            {
                "rewrite": "rewrite_query",
                "no_rewrite": "retrieve"
            }
        )
        
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "extract_citations")
        workflow.add_edge("extract_citations", "check_hallucination")
        workflow.add_edge("check_hallucination", "check_grounding")
        workflow.add_edge("check_grounding", "evaluate_confidence")
        
        # Confidence evaluation -> retry or continue
        workflow.add_conditional_edges(
            "evaluate_confidence",
            self._route_after_confidence,
            {
                "retry": "rewrite_query",
                "accept": "save_conversation"
            }
        )
        
        workflow.add_edge("save_conversation", "cache_response")
        workflow.add_edge("cache_response", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    # Routing functions
    def _route_after_cache(self, state: EnhancedRAGState) -> Literal["cached", "not_cached"]:
        """Route based on cache hit."""
        return "cached" if state.get("answer") else "not_cached"
    
    def _route_after_security(self, state: EnhancedRAGState) -> Literal["blocked", "safe"]:
        """Route based on security check."""
        return "blocked" if state.get("blocked") else "safe"
    
    def _route_after_classification(self, state: EnhancedRAGState) -> Literal["rewrite", "no_rewrite"]:
        """Route based on query classification."""
        if not settings.ENABLE_QUERY_REWRITE:
            return "no_rewrite"
        
        classification = state.get("query_classification", {})
        should_rewrite = classification.get("should_rewrite", False)
        return "rewrite" if should_rewrite else "no_rewrite"
    
    def _route_after_confidence(self, state: EnhancedRAGState) -> Literal["retry", "accept"]:
        """Route based on confidence score."""
        if not settings.ENABLE_RETRY_ON_LOW_CONFIDENCE:
            return "accept"
        
        if state.get("should_retry") and state.get("retry_count", 0) < settings.MAX_REWRITE_ATTEMPTS:
            return "retry"
        return "accept"
    
    # Node implementations
    def _check_cache(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Check Redis cache for existing response."""
        logger.info("Step 1: Checking cache")
        state["processing_steps"].append("check_cache")
        
        try:
            if self.cache.enabled:
                cached_response = self.cache.get(state["query"], state.get("filters"))
                if cached_response:
                    logger.info("Cache HIT - returning cached response")
                    state["answer"] = cached_response.answer
                    state["citations"] = cached_response.citations
                    state["confidence_score"] = cached_response.confidence_score
                    state["retrieved_documents"] = cached_response.retrieved_documents
                    return state
        except Exception as e:
            logger.error(f"Cache check error: {e}")
        
        return state
    
    def _security_check(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Check for prompt injection and security threats."""
        logger.info("Step 2: Security check")
        state["processing_steps"].append("security_check")
        
        if settings.ENABLE_PROMPT_INJECTION_DETECTION:
            injection_check = self.injection_detector.detect(state["query"])
            state["injection_check"] = injection_check
            
            if injection_check["is_injection"] and injection_check["risk_level"] in ["high", "medium"]:
                if settings.BLOCK_HIGH_RISK_QUERIES:
                    state["blocked"] = True
                    state["block_reason"] = f"Security risk detected: {injection_check['risk_level']}"
                    logger.warning(f"Query blocked: {state['block_reason']}")
        
        return state
    
    def _load_conversation(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Load conversation history if session_id provided."""
        logger.info("Step 3: Loading conversation context")
        state["processing_steps"].append("load_conversation")
        
        session_id = state.get("session_id")
        if settings.ENABLE_CONVERSATION_MEMORY and session_id:
            try:
                session = self.conversation_manager.get_or_create_session(
                    session_id,
                    max_turns=settings.MAX_CONVERSATION_TURNS,
                    context_window=settings.CONVERSATION_CONTEXT_WINDOW
                )
                state["conversation_context"] = session.get_context()
                logger.info(f"Loaded conversation context ({len(session.turns)} turns)")
            except Exception as e:
                logger.error(f"Error loading conversation: {e}")
                state["conversation_context"] = ""
        else:
            state["conversation_context"] = ""
        
        return state
    
    def _classify_query(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Classify query to determine processing strategy."""
        logger.info("Step 4: Classifying query")
        state["processing_steps"].append("classify_query")
        
        classification = self.query_classifier.classify(state["query"])
        state["query_classification"] = classification
        
        logger.info(f"Query classification: {classification}")
        return state
    
    def _rewrite_query(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Rewrite query for better retrieval."""
        logger.info("Step 5: Rewriting query")
        state["processing_steps"].append("rewrite_query")
        
        state["rewrite_attempts"] = state.get("rewrite_attempts", 0) + 1
        
        try:
            rewrite_result = self.query_rewriter.rewrite(
                state["query"],
                context=state.get("conversation_context")
            )
            
            if not rewrite_result.get("blocked") and rewrite_result.get("rewritten_query"):
                state["query"] = rewrite_result["rewritten_query"]
                state["query_rewritten"] = True
                logger.info(f"Query rewritten (attempt {state['rewrite_attempts']})")
        except Exception as e:
            logger.error(f"Query rewrite error: {e}")
        
        return state
    
    def _retrieve_documents(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Retrieve documents using hybrid search."""
        logger.info("Step 6: Retrieving documents")
        state["processing_steps"].append("retrieve")
        
        try:
            documents = self.vector_store.hybrid_search(
                query=state["query"],
                top_k=state["top_k"] * 2,
                alpha=settings.HYBRID_SEARCH_ALPHA,
                filters=state.get("filters")
            )
            state["retrieved_documents"] = documents
            logger.info(f"Retrieved {len(documents)} documents")
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            state["error"] = f"Retrieval error: {str(e)}"
            state["retrieved_documents"] = []
        
        return state
    
    def _rerank_documents(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Rerank retrieved documents."""
        logger.info("Step 7: Reranking documents")
        state["processing_steps"].append("rerank")
        
        try:
            if state["retrieved_documents"]:
                reranked = self.reranker.rerank(
                    query=state["query"],
                    documents=state["retrieved_documents"],
                    top_k=state["top_k"]
                )
                state["reranked_documents"] = reranked
                logger.info(f"Reranked to top {len(reranked)} documents")
            else:
                state["reranked_documents"] = []
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            state["reranked_documents"] = state["retrieved_documents"][:state["top_k"]]
        
        return state
    
    def _generate_answer(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Generate answer with conversation context."""
        logger.info("Step 8: Generating answer")
        state["processing_steps"].append("generate")
        
        try:
            if not state["reranked_documents"]:
                state["answer"] = "I don't have enough information to answer this question."
                state["confidence_score"] = 0.0
                return state
            
            # Prepare context
            context_parts = []
            for i, doc in enumerate(state["reranked_documents"], 1):
                context_parts.append(
                    f"[Source {i}] (from {doc.metadata.source})\n{doc.content}"
                )
            context = "\n\n".join(context_parts)
            state["context"] = context
            
            # Create prompt with conversation context
            prompt_template = """
You are a technical assistant providing accurate answers.

{conversation_context}

Context:
{context}

Question: {query}

Answer (with [Source N] citations):"""
            
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm
            
            response = chain.invoke({
                "conversation_context": state.get("conversation_context", ""),
                "context": context,
                "query": state["query"]
            })
            
            state["answer"] = response
            
            # Calculate confidence
            avg_score = sum(
                doc.rerank_score or doc.similarity_score or 0.0
                for doc in state["reranked_documents"]
            ) / len(state["reranked_documents"])
            state["confidence_score"] = avg_score
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            state["error"] = f"Generation error: {str(e)}"
            state["answer"] = "An error occurred while generating the answer."
            state["confidence_score"] = 0.0
        
        return state
    
    def _extract_citations(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Extract citations from answer."""
        logger.info("Step 9: Extracting citations")
        state["processing_steps"].append("extract_citations")
        
        try:
            import re
            citations = []
            citation_pattern = r'\[Source (\d+)\]'
            cited_sources = set(re.findall(citation_pattern, state["answer"]))
            
            for source_num in cited_sources:
                idx = int(source_num) - 1
                if idx < len(state["reranked_documents"]):
                    doc = state["reranked_documents"][idx]
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
        except Exception as e:
            logger.error(f"Citation extraction error: {e}")
            state["citations"] = []
        
        return state
    
    def _check_hallucination(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Check for hallucinations."""
        logger.info("Step 10: Checking hallucinations")
        state["processing_steps"].append("check_hallucination")
        
        try:
            if settings.ENABLE_HALLUCINATION_CHECK:
                hallucination_check = self.hallucination_detector.detect(
                    response=state["answer"],
                    context_documents=state["reranked_documents"]
                )
                state["hallucination_check"] = hallucination_check
                
                if hallucination_check.is_hallucinated:
                    state["confidence_score"] *= 0.5
        except Exception as e:
            logger.error(f"Hallucination check error: {e}")
        
        return state
    
    def _check_grounding(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Check context grounding."""
        logger.info("Step 11: Checking grounding")
        state["processing_steps"].append("check_grounding")
        
        try:
            grounding_metrics = self.grounding_checker.check_grounding(
                response=state["answer"],
                context_documents=state["reranked_documents"]
            )
            state["grounding_metrics"] = grounding_metrics
            
            if not grounding_metrics.get("is_well_grounded", True):
                state["confidence_score"] *= 0.7
        except Exception as e:
            logger.error(f"Grounding check error: {e}")
        
        return state
    
    def _evaluate_confidence(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Evaluate if retry is needed based on confidence."""
        logger.info("Step 12: Evaluating confidence")
        state["processing_steps"].append("evaluate_confidence")
        
        confidence = state.get("confidence_score", 0.0)
        retry_count = state.get("retry_count", 0)
        
        # Determine if retry is needed
        if (confidence < settings.RETRY_CONFIDENCE_THRESHOLD and 
            retry_count < settings.MAX_REWRITE_ATTEMPTS):
            state["should_retry"] = True
            state["retry_count"] = retry_count + 1
            logger.warning(f"Low confidence ({confidence:.2f}), triggering retry {state['retry_count']}")
        else:
            state["should_retry"] = False
        
        return state
    
    def _save_conversation(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Save conversation turn to memory."""
        logger.info("Step 13: Saving conversation")
        state["processing_steps"].append("save_conversation")
        
        session_id = state.get("session_id")
        if settings.ENABLE_CONVERSATION_MEMORY and session_id:
            try:
                session = self.conversation_manager.get_or_create_session(session_id)
                session.add_turn(
                    query=state["original_query"],
                    answer=state["answer"],
                    metadata={
                        "confidence": state["confidence_score"],
                        "rewritten": state.get("query_rewritten", False)
                    }
                )
            except Exception as e:
                logger.error(f"Error saving conversation: {e}")
        
        return state
    
    def _cache_response(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Cache the response."""
        logger.info("Step 14: Caching response")
        state["processing_steps"].append("cache_response")
        
        try:
            if self.cache.enabled and state.get("confidence_score", 0) > 0.7:
                response = QueryResponse(
                    answer=state["answer"],
                    citations=state["citations"],
                    retrieved_documents=state["reranked_documents"],
                    hallucination_check=state.get("hallucination_check"),
                    query_type=state["query_type"],
                    confidence_score=state["confidence_score"],
                    processing_time_ms=0.0,  # Will be updated
                    metadata={}
                )
                self.cache.set(state["original_query"], response, state.get("filters"))
        except Exception as e:
            logger.error(f"Caching error: {e}")
        
        return state
    
    def _handle_blocked(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Handle blocked queries."""
        logger.info("Handling blocked query")
        state["processing_steps"].append("handle_blocked")
        state["answer"] = f"Query blocked: {state.get('block_reason', 'Security check failed')}"
        state["confidence_score"] = 0.0
        return state
    
    def _finalize_response(self, state: EnhancedRAGState) -> EnhancedRAGState:
        """Finalize the response."""
        logger.info("Step 15: Finalizing response")
        state["processing_steps"].append("finalize")
        
        processing_time = (time.time() - state["start_time"]) * 1000
        logger.info(
            f"Pipeline complete in {processing_time:.2f}ms, "
            f"confidence: {state['confidence_score']:.2f}, "
            f"retries: {state.get('retry_count', 0)}"
        )
        
        return state
    
    def query(self, request: QueryRequest) -> QueryResponse:
        """Process query through enhanced RAG pipeline."""
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Initialize state
        initial_state: EnhancedRAGState = {
            "query": request.query,
            "original_query": request.query,
            "session_id": request.session_id,
            "query_type": request.query_type or QueryType.FACTUAL,
            "filters": request.filters or {},
            "top_k": request.top_k or settings.RERANK_TOP_K,
            "query_classification": {},
            "injection_check": {},
            "rewrite_attempts": 0,
            "query_rewritten": False,
            "retrieved_documents": [],
            "reranked_documents": [],
            "context": "",
            "answer": "",
            "citations": [],
            "conversation_context": "",
            "hallucination_check": None,
            "grounding_metrics": {},
            "confidence_score": 0.0,
            "processing_steps": [],
            "start_time": time.time(),
            "should_retry": False,
            "retry_count": 0,
            "error": "",
            "blocked": False,
            "block_reason": ""
        }
        
        try:
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Build response
            processing_time = (time.time() - final_state["start_time"]) * 1000
            
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
                    "query_rewritten": final_state.get("query_rewritten", False),
                    "retry_count": final_state.get("retry_count", 0),
                    "blocked": final_state.get("blocked", False),
                    "cache_hit": "check_cache" in final_state["processing_steps"] and len(final_state["processing_steps"]) == 2
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in enhanced RAG pipeline: {e}")
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