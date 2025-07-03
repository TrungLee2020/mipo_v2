import os
import pickle
import logging
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from retrieval.retrieval_qdrant import RetrievalResult

logger = logging.getLogger(__name__)
# Reranker imports
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logging.warning("sentence-transformers not available for reranking")


@dataclass
class HybridRetrievalResult(RetrievalResult):
    """Extended retrieval result with hybrid scoring information"""
    embedding_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: Optional[float] = None
    fusion_score: float = 0.0
    retrieval_source: str = "hybrid"  # "embedding", "bm25", "hybrid"

class CrossEncoderReranker:
    """Cross-encoder reranker for final result ranking"""
    
    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-reranker-base"):
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load cross-encoder model"""
        if not RERANKER_AVAILABLE:
            logger.warning("Reranker not available - sentence-transformers not installed")
            return
        
        try:
            self.model = CrossEncoder(self.model_name, trust_remote_code=True)
            logger.info(f"✅ Loaded reranker model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Error loading reranker: {e}")
            self.model = None
    
    def rerank(self, query: str, results: List[HybridRetrievalResult], top_k: int = None) -> List[HybridRetrievalResult]:
        """Rerank results using cross-encoder"""
        if not self.model or not results:
            return results
        
        try:
            # Prepare query-document pairs
            pairs = []
            for result in results:
                # Use chunk content for reranking
                content = result.chunk.content
                # Truncate if too long
                if len(content) > 2000:
                    content = content[:2000] + "..."
                pairs.append([query, content])
            
            # Get rerank scores
            rerank_scores = self.model.predict(pairs)
            
            # Update results with rerank scores
            for i, result in enumerate(results):
                result.rerank_score = float(rerank_scores[i])
            
            # Sort by rerank score
            reranked_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
            
            if top_k:
                reranked_results = reranked_results[:top_k]
            
            logger.info(f"✅ Reranked {len(results)} results")
            return reranked_results
            
        except Exception as e:
            logger.error(f"❌ Reranking error: {e}")
            return results
