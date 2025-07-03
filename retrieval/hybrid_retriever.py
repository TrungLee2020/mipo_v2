# retrieval/hybrid_retriever.py - Enhanced hybrid retrieval with BM25 and reranking

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from processing.chunker import Chunk
from processing.embedder import VietnameseEmbedder
from retrieval.retrieval_qdrant import RetrievalResult
from retrieval.vector_db_qdrant import QdrantVectorStore
from retrieval.bm25_store import BM25Manager
from retrieval.reranker import CrossEncoderReranker
from utils.utils import estimate_tokens

logger = logging.getLogger(__name__)

@dataclass
class HybridRetrievalResult(RetrievalResult):
    """Extended retrieval result with hybrid scoring information"""
    embedding_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: Optional[float] = None
    fusion_score: float = 0.0
    retrieval_source: str = "hybrid"  # "embedding", "bm25", "hybrid"

class HybridRetriever:
    """Enhanced hybrid retriever combining embedding, BM25, and reranking"""
    
    def __init__(self, 
                 vector_store: QdrantVectorStore,
                 embedder: VietnameseEmbedder,
                 bm25_path: str = "./indices/bm25",
                 reranker_model: str = "Alibaba-NLP/gte-multilingual-reranker-base",
                 enable_reranker: bool = True):
        
        self.vector_store = vector_store
        self.embedder = embedder
        self.enable_reranker = enable_reranker  # and RERANKER_AVAILABLE
        
        # Initialize BM25
        self.bm25_manager = BM25Manager(bm25_path)
        
        # Initialize reranker
        self.reranker = None
        if self.enable_reranker:
            self.reranker = CrossEncoderReranker(reranker_model)
        
        # Fusion weights
        self.embedding_weight = 0.7
        self.bm25_weight = 0.3
        
        logger.info(f"🔄 Initialized HybridRetriever (Reranker: {self.enable_reranker})")
    
    def build_bm25_index(self, chunks: List[Chunk]) -> bool:
        """Build BM25 index from chunks"""
        return self.bm25_manager.build_index(chunks)
    
    def load_bm25_index(self) -> bool:
        """Load existing BM25 index"""
        return self.bm25_manager.load_index()
    
    def retrieve(self, 
                query: str, 
                k: int = 5, 
                retrieval_strategy: str = 'hybrid_enhanced',
                embedding_weight: float = None,
                bm25_weight: float = None,
                enable_rerank: bool = None) -> List[HybridRetrievalResult]:
        """
        Main hybrid retrieval function
        
        Args:
            query: Search query
            k: Number of results to return
            retrieval_strategy: 'hybrid_enhanced', 'embedding_only', 'bm25_only', 'ensemble'
            embedding_weight: Weight for embedding scores (overrides default)
            bm25_weight: Weight for BM25 scores (overrides default)
            enable_rerank: Whether to use reranking (overrides default)
        """
        
        # Override weights if provided
        if embedding_weight is not None:
            self.embedding_weight = embedding_weight
        if bm25_weight is not None:
            self.bm25_weight = bm25_weight
        if enable_rerank is not None:
            use_rerank = enable_rerank and self.enable_reranker
        else:
            use_rerank = self.enable_reranker
        
        logger.info(f"🔍 Hybrid retrieval: {retrieval_strategy} (k={k}, rerank={use_rerank})")
        
        if retrieval_strategy == 'embedding_only':
            return self._embedding_only_retrieve(query, k, use_rerank)
        elif retrieval_strategy == 'bm25_only':
            return self._bm25_only_retrieve(query, k, use_rerank)
        elif retrieval_strategy == 'ensemble':
            return self._ensemble_retrieve(query, k, use_rerank)
        else:  # hybrid_enhanced (default)
            return self._hybrid_enhanced_retrieve(query, k, use_rerank)
    
    def _embedding_only_retrieve(self, query: str, k: int, use_rerank: bool) -> List[HybridRetrievalResult]:
        """Pure embedding-based retrieval"""
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, k * 2)
        
        # Convert to hybrid results
        hybrid_results = []
        for result in vector_results:
            hybrid_result = HybridRetrievalResult(
                chunk_id=result.chunk_id,
                score=result.score,
                chunk=result.chunk,
                parent_chunk=result.parent_chunk,
                related_chunks=result.related_chunks,
                context_metadata=result.context_metadata,
                embedding_score=result.score,
                bm25_score=0.0,
                fusion_score=result.score,
                retrieval_source="embedding"
            )
            hybrid_results.append(hybrid_result)
        
        # Rerank if enabled
        if use_rerank and self.reranker:
            hybrid_results = self.reranker.rerank(query, hybrid_results, k)
        else:
            hybrid_results = hybrid_results[:k]
        
        return hybrid_results
    
    def _bm25_only_retrieve(self, query: str, k: int, use_rerank: bool) -> List[HybridRetrievalResult]:
        """Pure BM25-based retrieval"""
        bm25_results = self.bm25_manager.search(query, k * 2)
        
        # Convert to hybrid results
        hybrid_results = []
        for chunk_id, bm25_score in bm25_results:
            chunk = self.bm25_manager.get_chunk_by_id(chunk_id)
            if chunk:
                # Get enhanced context from vector store
                # vector_chunk = self.vector_store.get_chunk_by_id(chunk_id)
                
                hybrid_result = HybridRetrievalResult(
                    chunk_id=chunk_id,
                    score=bm25_score,
                    chunk=chunk,
                    parent_chunk=None,  # Will be filled if available
                    related_chunks=[],
                    context_metadata={},
                    embedding_score=0.0,
                    bm25_score=bm25_score,
                    fusion_score=bm25_score,
                    retrieval_source="bm25"
                )
                hybrid_results.append(hybrid_result)
        
        # Rerank if enabled
        if use_rerank and self.reranker:
            hybrid_results = self.reranker.rerank(query, hybrid_results, k)
        else:
            hybrid_results = hybrid_results[:k]
        
        return hybrid_results
    
    def _ensemble_retrieve(self, query: str, k: int, use_rerank: bool) -> List[HybridRetrievalResult]:
        """Ensemble retrieval - combine results from both methods independently"""
        
        # Get embedding results
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, k)
        
        # Get BM25 results
        bm25_results = self.bm25_manager.search(query, k)
        
        # Combine results
        all_results = {}
        
        # Add embedding results
        for result in vector_results:
            hybrid_result = HybridRetrievalResult(
                chunk_id=result.chunk_id,
                score=result.score,
                chunk=result.chunk,
                parent_chunk=result.parent_chunk,
                related_chunks=result.related_chunks,
                context_metadata=result.context_metadata,
                embedding_score=result.score,
                bm25_score=0.0,
                fusion_score=result.score * self.embedding_weight,
                retrieval_source="embedding"
            )
            all_results[result.chunk_id] = hybrid_result
        
        # Add/merge BM25 results
        for chunk_id, bm25_score in bm25_results:
            if chunk_id in all_results:
                # Merge scores
                existing = all_results[chunk_id]
                existing.bm25_score = bm25_score
                existing.fusion_score = (existing.embedding_score * self.embedding_weight + 
                                       bm25_score * self.bm25_weight)
                existing.retrieval_source = "hybrid"
            else:
                # Add new BM25-only result
                chunk = self.bm25_manager.get_chunk_by_id(chunk_id)
                if chunk:
                    hybrid_result = HybridRetrievalResult(
                        chunk_id=chunk_id,
                        score=bm25_score,
                        chunk=chunk,
                        parent_chunk=None,
                        related_chunks=[],
                        context_metadata={},
                        embedding_score=0.0,
                        bm25_score=bm25_score,
                        fusion_score=bm25_score * self.bm25_weight,
                        retrieval_source="bm25"
                    )
                    all_results[chunk_id] = hybrid_result
        
        # Sort by fusion score
        sorted_results = sorted(all_results.values(), key=lambda x: x.fusion_score, reverse=True)
        
        # Rerank if enabled
        if use_rerank and self.reranker:
            sorted_results = self.reranker.rerank(query, sorted_results, k)
        else:
            sorted_results = sorted_results[:k]
        
        return sorted_results
    
    def _hybrid_enhanced_retrieve(self, query: str, k: int, use_rerank: bool) -> List[HybridRetrievalResult]:
        """Enhanced hybrid retrieval with RRF (Reciprocal Rank Fusion)"""
        
        # Get embedding results with higher k for fusion
        query_embedding = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, k * 3)
        
        # Get BM25 results
        bm25_results = self.bm25_manager.search(query, k * 3)
        
        # Create rank maps
        embedding_ranks = {result.chunk_id: i + 1 for i, result in enumerate(vector_results)}
        bm25_ranks = {chunk_id: i + 1 for i, (chunk_id, _) in enumerate(bm25_results)}
        bm25_scores = {chunk_id: score for chunk_id, score in bm25_results}
        
        # Collect all unique chunk IDs
        all_chunk_ids = set(embedding_ranks.keys()) | set(bm25_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        rrf_constant = 60  # Standard RRF constant
        
        for chunk_id in all_chunk_ids:
            rrf_score = 0.0
            
            # Add embedding contribution
            if chunk_id in embedding_ranks:
                rrf_score += self.embedding_weight / (rrf_constant + embedding_ranks[chunk_id])
            
            # Add BM25 contribution
            if chunk_id in bm25_ranks:
                rrf_score += self.bm25_weight / (rrf_constant + bm25_ranks[chunk_id])
            
            rrf_scores[chunk_id] = rrf_score
        
        # Sort by RRF score and get top candidates
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_chunk_ids = sorted_chunk_ids[:k * 2]  # Get more for reranking
        
        # Create hybrid results
        hybrid_results = []
        for chunk_id in top_chunk_ids:
            # Try to get chunk from vector store first (has enhanced context)
            vector_result = next((r for r in vector_results if r.chunk_id == chunk_id), None)
            
            if vector_result:
                chunk = vector_result.chunk
                parent_chunk = vector_result.parent_chunk
                related_chunks = vector_result.related_chunks
                context_metadata = vector_result.context_metadata
                embedding_score = vector_result.score
            else:
                # Get from BM25 manager
                chunk = self.bm25_manager.get_chunk_by_id(chunk_id)
                if not chunk:
                    continue
                parent_chunk = None
                related_chunks = []
                context_metadata = {}
                embedding_score = 0.0
            
            hybrid_result = HybridRetrievalResult(
                chunk_id=chunk_id,
                score=rrf_scores[chunk_id],
                chunk=chunk,
                parent_chunk=parent_chunk,
                related_chunks=related_chunks,
                context_metadata=context_metadata,
                embedding_score=embedding_score,
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                fusion_score=rrf_scores[chunk_id],
                retrieval_source="hybrid"
            )
            
            # Add retrieval method info to metadata
            hybrid_result.context_metadata.update({
                'embedding_rank': embedding_ranks.get(chunk_id, 999),
                'bm25_rank': bm25_ranks.get(chunk_id, 999),
                'rrf_score': rrf_scores[chunk_id],
                'retrieval_methods': {
                    'embedding': chunk_id in embedding_ranks,
                    'bm25': chunk_id in bm25_ranks
                }
            })
            
            hybrid_results.append(hybrid_result)
        
        # Rerank if enabled
        if use_rerank and self.reranker:
            hybrid_results = self.reranker.rerank(query, hybrid_results, k)
        else:
            hybrid_results = hybrid_results[:k]
        
        return hybrid_results
    
    def format_context(self, results: List[HybridRetrievalResult], max_tokens: int = 4096) -> str:
        """Format hybrid retrieval results for context"""
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(results):
            if current_tokens >= max_tokens:
                break
            
            # Create enhanced context block with hybrid info
            context_block = self._create_hybrid_context_block(result, i + 1)
            block_tokens = estimate_tokens(context_block)
            
            if current_tokens + block_tokens > max_tokens:
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 300:
                    truncated_block = context_block[:int(remaining_tokens * 4)]  # Rough char estimate
                    context_parts.append(truncated_block + "\n[... truncated ...]")
                break
            
            context_parts.append(context_block)
            current_tokens += block_tokens
        
        return "\n\n" + "="*80 + "\n\n".join(context_parts)
    
    def _create_hybrid_context_block(self, result: HybridRetrievalResult, source_number: int) -> str:
        """Create context block with hybrid scoring information"""
        block_parts = []
        
        # Enhanced header with hybrid scores
        header_parts = [
            f"**📄 Nguồn {source_number}** (Fusion: {result.fusion_score:.4f})",
            f"**🔍 Điểm chi tiết:** Embedding: {result.embedding_score:.3f} | BM25: {result.bm25_score:.3f}"
        ]
        
        if result.rerank_score is not None:
            header_parts.append(f"**🎯 Rerank:** {result.rerank_score:.3f}")
        
        header_parts.append(f"**📊 Nguồn:** {result.retrieval_source}")
        
        # Add retrieval method info
        retrieval_methods = result.context_metadata.get('retrieval_methods', {})
        if retrieval_methods:
            methods = []
            if retrieval_methods.get('embedding'):
                rank = result.context_metadata.get('embedding_rank', 999)
                methods.append(f"Embedding (#{rank})")
            if retrieval_methods.get('bm25'):
                rank = result.context_metadata.get('bm25_rank', 999)
                methods.append(f"BM25 (#{rank})")
            if methods:
                header_parts.append(f"**🔄 Phương pháp:** {' + '.join(methods)}")
        
        # Document info
        metadata = result.chunk.metadata
        doc_title = metadata.get('document_title', 'Không rõ')
        header_parts.append(f"**📋 Tài liệu:** {doc_title}")
        
        # Section info
        section_title = metadata.get('section_title', '')
        if section_title:
            header_parts.append(f"**📍 Phần:** {section_title}")
        
        block_parts.append("\n".join(header_parts))
        
        # Main content
        content_preview = result.chunk.content
        if len(content_preview) > 800:
            content_preview = content_preview[:800] + "..."
        
        block_parts.append(f"**📄 Nội dung:**\n{content_preview}")
        
        # Parent context if available
        if result.parent_chunk:
            parent_preview = result.parent_chunk.content[:300] + "..." if len(result.parent_chunk.content) > 300 else result.parent_chunk.content
            block_parts.append(f"**👆 Bối cảnh phần chính:**\n{parent_preview}")
        
        return "\n\n".join(block_parts)
    
    def get_stats(self) -> Dict:
        """Get hybrid retriever statistics"""
        stats = {
            'vector_store_stats': self.vector_store.get_stats(),
            'bm25_available': self.bm25_manager.retriever is not None,
            'bm25_chunks': len(self.bm25_manager.chunk_map),
            'reranker_available': self.enable_reranker,
            'reranker_model': self.reranker.model_name if self.reranker else None,
            'fusion_weights': {
                'embedding': self.embedding_weight,
                'bm25': self.bm25_weight
            }
        }
        return stats