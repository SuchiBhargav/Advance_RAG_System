"""
Query rewriting and optimization module for improving retrieval quality.
Includes query expansion, clarification, and prompt injection defense.
"""

import re
from typing import List, Optional, Dict, Any
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptInjectionDetector:
    """
    Detects and prevents prompt injection attacks.
    """
    
    # Patterns that indicate potential prompt injection
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions?",
        r"disregard\s+(previous|above|all)\s+instructions?",
        r"forget\s+(previous|above|all)\s+instructions?",
        r"new\s+instructions?:",
        r"system\s*:",
        r"<\s*system\s*>",
        r"you\s+are\s+now",
        r"act\s+as\s+(a|an)",
        r"pretend\s+(you|to)\s+are",
        r"roleplay\s+as",
        r"simulate\s+(a|an)",
        r"\[INST\]",
        r"\[/INST\]",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"###\s*Instruction",
        r"###\s*System",
        r"jailbreak",
        r"DAN\s+mode",
    ]
    
    def __init__(self):
        """Initialize prompt injection detector."""
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS]
        logger.info("Initialized PromptInjectionDetector")
    
    def detect(self, query: str) -> Dict[str, Any]:
        """
        Detect potential prompt injection in query.
        
        Args:
            query: User query to check
        
        Returns:
            Dictionary with detection results
        """
        detected_patterns = []
        
        for pattern in self.patterns:
            matches = pattern.findall(query)
            if matches:
                detected_patterns.append({
                    "pattern": pattern.pattern,
                    "matches": matches
                })
        
        is_injection = len(detected_patterns) > 0
        
        if is_injection:
            logger.warning(f"Potential prompt injection detected: {detected_patterns}")
        
        return {
            "is_injection": is_injection,
            "confidence": min(len(detected_patterns) * 0.3, 1.0),
            "detected_patterns": detected_patterns,
            "risk_level": self._calculate_risk_level(len(detected_patterns))
        }
    
    def _calculate_risk_level(self, pattern_count: int) -> str:
        """Calculate risk level based on pattern count."""
        if pattern_count == 0:
            return "none"
        elif pattern_count == 1:
            return "low"
        elif pattern_count <= 3:
            return "medium"
        else:
            return "high"
    
    def sanitize(self, query: str) -> str:
        """
        Sanitize query by removing potential injection patterns.
        
        Args:
            query: User query
        
        Returns:
            Sanitized query
        """
        sanitized = query
        
        for pattern in self.patterns:
            sanitized = pattern.sub("[REMOVED]", sanitized)
        
        # Remove excessive special characters
        sanitized = re.sub(r'[<>{}[\]]{3,}', '', sanitized)
        
        return sanitized.strip()


class QueryRewriter:
    """
    Rewrites and optimizes queries for better retrieval.
    """
    
    def __init__(self):
        """Initialize query rewriter with LLM."""
        self.llm = OllamaLLM(
            model=settings.LLM_MODEL,
            temperature=0.3  # Lower temperature for more focused rewrites
        )
        self.injection_detector = PromptInjectionDetector()
        logger.info("Initialized QueryRewriter")
    
    def rewrite(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Rewrite query for better retrieval.
        
        Args:
            query: Original user query
            context: Optional conversation context
        
        Returns:
            Dictionary with rewritten query and metadata
        """
        # First check for prompt injection
        injection_check = self.injection_detector.detect(query)
        
        if injection_check["is_injection"] and injection_check["risk_level"] in ["high", "medium"]:
            logger.warning(f"Blocking query due to injection risk: {injection_check['risk_level']}")
            return {
                "original_query": query,
                "rewritten_query": None,
                "blocked": True,
                "reason": "Potential prompt injection detected",
                "injection_check": injection_check
            }
        
        try:
            # Rewrite query for clarity and specificity
            rewritten = self._rewrite_with_llm(query, context)
            
            # Generate query variations for better retrieval
            variations = self._generate_variations(rewritten)
            
            return {
                "original_query": query,
                "rewritten_query": rewritten,
                "variations": variations,
                "blocked": False,
                "injection_check": injection_check
            }
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return {
                "original_query": query,
                "rewritten_query": query,  # Fall back to original
                "variations": [query],
                "blocked": False,
                "error": str(e)
            }
    
    def _rewrite_with_llm(self, query: str, context: Optional[str] = None) -> str:
        """
        Rewrite query using LLM for clarity.
        
        Args:
            query: Original query
            context: Optional context
        
        Returns:
            Rewritten query
        """
        prompt = ChatPromptTemplate.from_template("""
You are a query optimization expert. Rewrite the user's query to be more specific and clear for document retrieval.

Rules:
1. Keep the core intent of the query
2. Make it more specific and detailed
3. Expand abbreviations and acronyms
4. Add relevant technical terms if applicable
5. Keep it concise (max 2 sentences)
6. Do NOT change the meaning

{context_section}

Original Query: {query}

Rewritten Query:""")
        
        context_section = f"Previous Context: {context}\n" if context else ""
        
        chain = prompt | self.llm
        rewritten = chain.invoke({
            "query": query,
            "context_section": context_section
        })
        
        # Clean up the response
        rewritten = rewritten.strip()
        
        # If rewrite is too different or too long, use original
        if len(rewritten) > len(query) * 3 or len(rewritten) < 5:
            logger.debug("Rewrite rejected, using original query")
            return query
        
        logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
        return rewritten
    
    def _generate_variations(self, query: str) -> List[str]:
        """
        Generate query variations for multi-query retrieval.
        
        Args:
            query: Base query
        
        Returns:
            List of query variations
        """
        variations = [query]
        
        try:
            prompt = ChatPromptTemplate.from_template("""
Generate 2 alternative phrasings of this query that preserve the same meaning but use different words.

Original: {query}

Alternative 1:
Alternative 2:""")
            
            chain = prompt | self.llm
            response = chain.invoke({"query": query})
            
            # Parse alternatives
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Remove "Alternative N:" prefix if present
                line = re.sub(r'^Alternative\s+\d+:\s*', '', line, flags=re.IGNORECASE)
                if line and line != query and len(line) > 5:
                    variations.append(line)
            
            # Limit to 3 total variations
            variations = variations[:3]
            
        except Exception as e:
            logger.error(f"Error generating variations: {e}")
        
        return variations
    
    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
        
        Returns:
            Expanded query
        """
        # Simple expansion with common technical synonyms
        expansions = {
            "error": "error issue problem bug",
            "fix": "fix resolve solution workaround",
            "how": "how method way approach",
            "why": "why reason cause explanation",
            "install": "install setup configure deployment",
            "api": "api endpoint interface service",
            "database": "database db storage data",
            "server": "server backend service application",
        }
        
        expanded = query
        for term, expansion in expansions.items():
            if term in query.lower():
                expanded = expanded + " " + expansion
        
        return expanded.strip()


class QueryClassifier:
    """
    Classifies queries to determine optimal retrieval strategy.
    """
    
    def __init__(self):
        """Initialize query classifier."""
        logger.info("Initialized QueryClassifier")
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify query characteristics.
        
        Args:
            query: User query
        
        Returns:
            Classification results
        """
        query_lower = query.lower()
        
        # Determine if query is vague
        vague_indicators = ["something", "anything", "stuff", "thing", "general", "overview"]
        is_vague = any(indicator in query_lower for indicator in vague_indicators)
        
        # Determine if query needs clarification
        needs_clarification = len(query.split()) < 3 or is_vague
        
        # Determine complexity
        complexity = "simple" if len(query.split()) < 5 else "moderate" if len(query.split()) < 15 else "complex"
        
        # Determine if it's a question
        is_question = query.strip().endswith('?') or any(
            query_lower.startswith(q) for q in ["what", "how", "why", "when", "where", "who", "which"]
        )
        
        return {
            "is_vague": is_vague,
            "needs_clarification": needs_clarification,
            "complexity": complexity,
            "is_question": is_question,
            "word_count": len(query.split()),
            "should_rewrite": is_vague or needs_clarification
        }


# Made with Bob