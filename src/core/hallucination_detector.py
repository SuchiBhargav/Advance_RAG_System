"""
Hallucination detection module for identifying unsupported claims in generated responses.
Uses multiple techniques: entailment checking, fact verification, and context grounding.
"""

from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import re

from config.settings import settings
from src.utils.logger import get_logger
from src.models.schemas import HallucinationCheck, RetrievedDocument

logger = get_logger(__name__)


class HallucinationDetector:
    """
    Detects hallucinations in generated responses by checking if claims
    are supported by the retrieved context.
    """
    
    def __init__(self):
        """Initialize hallucination detector with NLI model."""
        try:
            # Use Natural Language Inference (NLI) model for entailment checking
            logger.info("Loading NLI model for hallucination detection")
            self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
            logger.info("NLI model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading NLI model: {e}")
            self.nli_model = None
    
    def detect(
        self,
        response: str,
        context_documents: List[RetrievedDocument],
        threshold: Optional[float] = None
    ) -> HallucinationCheck:
        """
        Detect hallucinations in the response.
        
        Args:
            response: Generated response text
            context_documents: Retrieved context documents
            threshold: Confidence threshold for hallucination detection
        
        Returns:
            HallucinationCheck object with detection results
        """
        threshold = threshold or settings.HALLUCINATION_THRESHOLD
        
        if not self.nli_model:
            logger.warning("NLI model not available, skipping hallucination detection")
            return HallucinationCheck(
                is_hallucinated=False,
                confidence=0.0,
                explanation="Hallucination detection unavailable"
            )
        
        try:
            # Extract claims from response
            claims = self._extract_claims(response)
            
            if not claims:
                logger.debug("No claims extracted from response")
                return HallucinationCheck(
                    is_hallucinated=False,
                    confidence=1.0,
                    explanation="No specific claims to verify"
                )
            
            # Combine all context
            context = "\n\n".join([doc.content for doc in context_documents])
            
            if not context.strip():
                logger.warning("No context available for verification")
                return HallucinationCheck(
                    is_hallucinated=True,
                    confidence=1.0,
                    explanation="No context available to verify claims",
                    unsupported_claims=claims
                )
            
            # Check each claim against context
            unsupported_claims = []
            entailment_scores = []
            
            for claim in claims:
                score = self._check_entailment(claim, context)
                entailment_scores.append(score)
                
                if score < threshold:
                    unsupported_claims.append(claim)
            
            # Calculate overall confidence
            avg_score = sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0.0
            
            # Determine if hallucinated
            is_hallucinated = len(unsupported_claims) > 0
            
            explanation = self._generate_explanation(
                is_hallucinated,
                len(claims),
                len(unsupported_claims),
                avg_score
            )
            
            result = HallucinationCheck(
                is_hallucinated=is_hallucinated,
                confidence=avg_score,
                explanation=explanation,
                unsupported_claims=unsupported_claims
            )
            
            logger.info(
                f"Hallucination detection: {is_hallucinated}, "
                f"confidence: {avg_score:.2f}, "
                f"unsupported: {len(unsupported_claims)}/{len(claims)}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in hallucination detection: {e}")
            return HallucinationCheck(
                is_hallucinated=False,
                confidence=0.0,
                explanation=f"Error during detection: {str(e)}"
            )
    
    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract individual claims from text.
        Splits text into sentences as individual claims.
        
        Args:
            text: Input text
        
        Returns:
            List of claims
        """
        # Split by sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out very short sentences and questions
            if len(sentence) > 10 and not sentence.endswith('?'):
                claims.append(sentence)
        
        return claims
    
    def _check_entailment(self, claim: str, context: str) -> float:
        """
        Check if claim is entailed by context using NLI model.
        
        Args:
            claim: Claim to verify
            context: Context to check against
        
        Returns:
            Entailment score (0-1, higher means more supported)
        """
        try:
            # Prepare input for NLI model
            # Format: [premise, hypothesis]
            pairs = [[context, claim]]
            
            # Get predictions
            # NLI models typically output: [contradiction, neutral, entailment]
            scores = self.nli_model.predict(pairs)
            
            # Convert to probability using softmax
            import numpy as np
            probs = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
            
            # Return entailment probability (last class)
            entailment_score = float(probs[0][-1])
            
            return entailment_score
            
        except Exception as e:
            logger.error(f"Error checking entailment: {e}")
            return 0.5  # Return neutral score on error
    
    def _generate_explanation(
        self,
        is_hallucinated: bool,
        total_claims: int,
        unsupported_claims: int,
        avg_score: float
    ) -> str:
        """
        Generate human-readable explanation of detection result.
        
        Args:
            is_hallucinated: Whether hallucination was detected
            total_claims: Total number of claims
            unsupported_claims: Number of unsupported claims
            avg_score: Average entailment score
        
        Returns:
            Explanation string
        """
        if not is_hallucinated:
            return (
                f"All {total_claims} claims are well-supported by the context "
                f"(avg confidence: {avg_score:.2f})"
            )
        else:
            return (
                f"{unsupported_claims} out of {total_claims} claims lack sufficient "
                f"support in the context (avg confidence: {avg_score:.2f})"
            )


class ContextGroundingChecker:
    """
    Additional checker to ensure response stays grounded in provided context.
    Uses simpler heuristics for faster checking.
    """
    
    def __init__(self):
        """Initialize context grounding checker."""
        logger.info("Initialized ContextGroundingChecker")
    
    def check_grounding(
        self,
        response: str,
        context_documents: List[RetrievedDocument]
    ) -> Dict[str, Any]:
        """
        Check if response is grounded in context using heuristics.
        
        Args:
            response: Generated response
            context_documents: Retrieved context
        
        Returns:
            Dictionary with grounding metrics
        """
        try:
            # Combine context
            context = " ".join([doc.content.lower() for doc in context_documents])
            response_lower = response.lower()
            
            # Extract key terms from response (simple tokenization)
            response_terms = set(re.findall(r'\b\w+\b', response_lower))
            response_terms = {t for t in response_terms if len(t) > 3}  # Filter short words
            
            # Check how many terms appear in context
            grounded_terms = {t for t in response_terms if t in context}
            
            grounding_ratio = len(grounded_terms) / len(response_terms) if response_terms else 0.0
            
            # Check for "I don't know" patterns
            uncertainty_patterns = [
                "i don't know",
                "i'm not sure",
                "cannot determine",
                "insufficient information",
                "not mentioned",
                "unclear"
            ]
            
            has_uncertainty = any(pattern in response_lower for pattern in uncertainty_patterns)
            
            result = {
                "grounding_ratio": grounding_ratio,
                "grounded_terms": len(grounded_terms),
                "total_terms": len(response_terms),
                "has_uncertainty_markers": has_uncertainty,
                "is_well_grounded": grounding_ratio > 0.6 or has_uncertainty
            }
            
            logger.debug(f"Grounding check: {grounding_ratio:.2f} ratio, {len(grounded_terms)}/{len(response_terms)} terms")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in grounding check: {e}")
            return {
                "grounding_ratio": 0.0,
                "is_well_grounded": False,
                "error": str(e)
            }

# Made with Bob
