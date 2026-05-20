"""
RAGAS-based evaluation framework for RAG system quality assessment.
Evaluates faithfulness, answer relevancy, context precision, and context recall.
"""

from typing import List, Dict, Any
from datasets import Dataset
import pandas as pd

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from config.settings import settings
from src.utils.logger import get_logger
from src.models.schemas import QueryRequest, QueryResponse

logger = get_logger(__name__)


class RAGASEvaluator:
    """
    Evaluates RAG system quality using RAGAS framework.
    Provides comprehensive metrics for system performance.
    """
    
    def __init__(self):
        """Initialize RAGAS evaluator."""
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ]
        logger.info("Initialized RAGASEvaluator")
    
    def evaluate_responses(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate RAG responses using RAGAS metrics.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of context lists (retrieved documents)
            ground_truths: Optional list of ground truth answers
        
        Returns:
            Dictionary of metric scores
        """
        try:
            logger.info(f"Evaluating {len(questions)} responses with RAGAS")
            
            # Prepare dataset
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts
            }
            
            if ground_truths:
                data["ground_truth"] = ground_truths
            
            dataset = Dataset.from_dict(data)
            
            # Run evaluation
            results = evaluate(
                dataset,
                metrics=self.metrics
            )
            
            # Extract scores
            scores = {
                "faithfulness": results["faithfulness"],
                "answer_relevancy": results["answer_relevancy"],
                "context_precision": results["context_precision"],
                "context_recall": results["context_recall"] if ground_truths else 0.0,
                "overall_score": self._calculate_overall_score(results)
            }
            
            logger.info(f"Evaluation complete: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "overall_score": 0.0,
                "error": str(e)
            }
    
    def evaluate_single_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG response.
        
        Args:
            question: Question text
            answer: Generated answer
            contexts: Retrieved context documents
            ground_truth: Optional ground truth answer
        
        Returns:
            Dictionary of metric scores
        """
        ground_truths = [ground_truth] if ground_truth else None
        
        return self.evaluate_responses(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=ground_truths
        )
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall quality score from individual metrics.
        
        Args:
            results: RAGAS evaluation results
        
        Returns:
            Overall score (0-1)
        """
        # Weighted average of metrics
        weights = {
            "faithfulness": 0.4,
            "answer_relevancy": 0.3,
            "context_precision": 0.2,
            "context_recall": 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in results:
                score += results[metric] * weight
        
        return score
    
    def batch_evaluate(
        self,
        evaluation_data: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Batch evaluate multiple responses and return detailed results.
        
        Args:
            evaluation_data: List of dicts with question, answer, contexts, ground_truth
        
        Returns:
            DataFrame with detailed evaluation results
        """
        try:
            logger.info(f"Batch evaluating {len(evaluation_data)} responses")
            
            # Extract data
            questions = [d["question"] for d in evaluation_data]
            answers = [d["answer"] for d in evaluation_data]
            contexts = [d["contexts"] for d in evaluation_data]
            ground_truths = [d.get("ground_truth") for d in evaluation_data]
            
            # Filter out None ground truths
            has_ground_truth = any(gt is not None for gt in ground_truths)
            if not has_ground_truth:
                ground_truths = None
            
            # Evaluate
            scores = self.evaluate_responses(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths
            )
            
            # Create detailed results DataFrame
            results_df = pd.DataFrame(evaluation_data)
            for metric, score in scores.items():
                results_df[metric] = score
            
            logger.info("Batch evaluation complete")
            return results_df
            
        except Exception as e:
            logger.error(f"Error in batch evaluation: {e}")
            return pd.DataFrame()
    
    def generate_evaluation_report(
        self,
        results: Dict[str, float],
        output_path: str = None
    ) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            results: Evaluation results
            output_path: Optional path to save report
        
        Returns:
            Report text
        """
        report = f"""
# RAG System Evaluation Report

## Overall Performance
- **Overall Score**: {results.get('overall_score', 0.0):.2%}

## Detailed Metrics

### Faithfulness: {results.get('faithfulness', 0.0):.2%}
Measures how factually accurate the answer is based on the context.
- Score > 0.8: Excellent - Answer is highly faithful to context
- Score 0.6-0.8: Good - Minor inconsistencies
- Score < 0.6: Needs improvement - Significant hallucinations

### Answer Relevancy: {results.get('answer_relevancy', 0.0):.2%}
Measures how relevant the answer is to the question.
- Score > 0.8: Excellent - Highly relevant answer
- Score 0.6-0.8: Good - Mostly relevant
- Score < 0.6: Needs improvement - Off-topic or incomplete

### Context Precision: {results.get('context_precision', 0.0):.2%}
Measures how precise the retrieved context is.
- Score > 0.8: Excellent - Highly relevant context
- Score 0.6-0.8: Good - Some irrelevant context
- Score < 0.6: Needs improvement - Too much noise

### Context Recall: {results.get('context_recall', 0.0):.2%}
Measures how much of the required information was retrieved.
- Score > 0.8: Excellent - All necessary info retrieved
- Score 0.6-0.8: Good - Most info retrieved
- Score < 0.6: Needs improvement - Missing key information

## Recommendations

"""
        
        # Add recommendations based on scores
        if results.get('faithfulness', 0.0) < 0.6:
            report += "- ⚠️ Improve hallucination detection and control\n"
        
        if results.get('answer_relevancy', 0.0) < 0.6:
            report += "- ⚠️ Improve query understanding and answer generation\n"
        
        if results.get('context_precision', 0.0) < 0.6:
            report += "- ⚠️ Improve retrieval and reranking quality\n"
        
        if results.get('context_recall', 0.0) < 0.6:
            report += "- ⚠️ Increase retrieval top-k or improve chunking strategy\n"
        
        if all(results.get(m, 0.0) >= 0.8 for m in ['faithfulness', 'answer_relevancy', 'context_precision']):
            report += "- ✅ System performing excellently across all metrics!\n"
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report


class ContinuousEvaluator:
    """
    Continuous evaluation system for monitoring RAG quality over time.
    """
    
    def __init__(self, evaluator: RAGASEvaluator = None):
        """Initialize continuous evaluator."""
        self.evaluator = evaluator or RAGASEvaluator()
        self.evaluation_history: List[Dict[str, Any]] = []
        logger.info("Initialized ContinuousEvaluator")
    
    def add_evaluation(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> None:
        """
        Add a new evaluation to the history.
        
        Args:
            question: Question text
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Optional ground truth
        """
        try:
            scores = self.evaluator.evaluate_single_response(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth
            )
            
            self.evaluation_history.append({
                "timestamp": pd.Timestamp.now(),
                "question": question,
                "answer": answer,
                "scores": scores
            })
            
            logger.debug(f"Added evaluation: {scores}")
            
        except Exception as e:
            logger.error(f"Error adding evaluation: {e}")
    
    def get_recent_performance(self, n: int = 100) -> Dict[str, float]:
        """
        Get average performance over recent evaluations.
        
        Args:
            n: Number of recent evaluations to consider
        
        Returns:
            Average scores
        """
        if not self.evaluation_history:
            return {}
        
        recent = self.evaluation_history[-n:]
        
        # Calculate averages
        avg_scores = {}
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'overall_score']:
            scores = [e['scores'].get(metric, 0.0) for e in recent]
            avg_scores[metric] = sum(scores) / len(scores) if scores else 0.0
        
        return avg_scores
    
    def detect_performance_degradation(
        self,
        threshold: float = 0.1,
        window: int = 50
    ) -> bool:
        """
        Detect if performance has degraded significantly.
        
        Args:
            threshold: Degradation threshold
            window: Window size for comparison
        
        Returns:
            True if degradation detected
        """
        if len(self.evaluation_history) < window * 2:
            return False
        
        # Compare recent vs previous window
        recent = self.get_recent_performance(window)
        previous_start = len(self.evaluation_history) - window * 2
        previous_end = len(self.evaluation_history) - window
        
        previous_scores = []
        for e in self.evaluation_history[previous_start:previous_end]:
            previous_scores.append(e['scores'].get('overall_score', 0.0))
        
        previous_avg = sum(previous_scores) / len(previous_scores) if previous_scores else 0.0
        recent_avg = recent.get('overall_score', 0.0)
        
        degradation = previous_avg - recent_avg
        
        if degradation > threshold:
            logger.warning(
                f"Performance degradation detected: "
                f"{previous_avg:.2f} -> {recent_avg:.2f} "
                f"(drop of {degradation:.2f})"
            )
            return True
        
        return False

# Made with Bob
