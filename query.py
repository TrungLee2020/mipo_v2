# query.py - Search and Retrieval System

from typing import List, Dict, Optional
from core.config import RAGConfig, create_vector_store, ConfigPresets
from processing.embedder import VietnameseEmbedder
from retrieval.hybrid_retriever import HybridRetriever, HybridRetrievalResult
import os
import json
import logging
import argparse
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentQuerySystem:
    """Enhanced system for querying indexed documents with hybrid search capabilities"""
    
    def __init__(self, config: RAGConfig = None, index_path: str = None, enable_hybrid: bool = True):
        self.config = config or RAGConfig()
        self.index_path = index_path or "./indices/default_index"
        self.enable_hybrid = enable_hybrid
        
        # Initialize components
        self.embedder = VietnameseEmbedder(self.config.embedding)
        self.vector_store = create_vector_store(self.config)
        
        # Initialize retriever
        if self.enable_hybrid:
            bm25_path = f"{self.index_path}_bm25"
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                embedder=self.embedder,
                bm25_path=bm25_path,
                enable_reranker=True
            )
            logger.info("🔍 Initialized Query System with Hybrid Retrieval")
        else:
            # Fallback to standard retriever
            from retrieval.retrieval_qdrant import HierarchicalRetriever
            self.retriever = HierarchicalRetriever(self.vector_store, self.embedder)
            logger.info("📊 Initialized standard Query System")
        
        # Load existing index
        self._load_index()
        
        # Query statistics
        self.query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_response_time': 0.0,
            'query_history': []
        }
    
    def _load_index(self):
        """Load existing vector index and BM25 index"""
        try:
            # Load vector store
            success = self.vector_store.load_index(self.index_path)
            if success:
                logger.info(f"✅ Loaded vector index from {self.index_path}")
                
                # Load BM25 index if hybrid mode
                if self.enable_hybrid and self.retriever:
                    bm25_success = self.retriever.load_bm25_index()
                    if bm25_success:
                        logger.info("✅ Loaded BM25 index")
                    else:
                        logger.warning("⚠️ BM25 index not found - hybrid search may be limited")
                
                # Load processing stats if available
                stats_path = f"{self.index_path}.stats"
                if os.path.exists(stats_path):
                    with open(stats_path, 'r', encoding='utf-8') as f:
                        index_stats = json.load(f)
                        logger.info(f"📊 Index contains {index_stats.get('documents_processed', 0)} documents")
                
                vector_stats = self.vector_store.get_stats()
                logger.info(f"📊 Vector store: {vector_stats.get('active_chunks', 0)} chunks available")
                
            else:
                logger.error(f"❌ Could not load vector index from {self.index_path}")
                logger.info("💡 Please run indexing.py first to create the index")
                
        except Exception as e:
            logger.error(f"❌ Error loading index: {e}")
    
    def query(self, 
              question: str, 
              k: int = 5, 
              strategy: str = 'hybrid_enhanced',
              embedding_weight: float = 0.7,
              bm25_weight: float = 0.3,
              enable_rerank: bool = True,
              include_metadata: bool = True) -> Dict:
        """Execute search query with hybrid retrieval"""
        
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing query: '{question[:50]}...' with strategy '{strategy}'")
            
            # Execute retrieval based on system configuration
            if self.enable_hybrid and self.retriever:
                results = self.retriever.retrieve(
                    query=question,
                    k=k,
                    retrieval_strategy=strategy,
                    embedding_weight=embedding_weight,
                    bm25_weight=bm25_weight,
                    enable_rerank=enable_rerank
                )
            else:
                # Fallback to standard retrieval
                results = self.retriever.retrieve(
                    query=question,
                    k=k,
                    retrieval_strategy='enhanced_hierarchical'
                )
            
            processing_time = time.time() - start_time
            
            # Handle empty results
            if not results:
                return self._create_empty_result(question, strategy, processing_time)
            
            # Format comprehensive results
            query_result = self._format_query_results(
                question, results, strategy, processing_time,
                embedding_weight, bm25_weight, enable_rerank, include_metadata
            )
            
            # Update statistics
            self._update_query_stats(question, processing_time, True)
            
            logger.info(f"✅ Query completed in {processing_time:.3f}s - Found {len(results)} results")
            return query_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Query error: {e}")
            
            self._update_query_stats(question, processing_time, False)
            
            return {
                'query': question,
                'answer': f'Lỗi khi xử lý câu hỏi: {str(e)}',
                'results': [],
                'context': '',
                'confidence': 0.0,
                'processing_time': processing_time,
                'error': str(e),
                'search_info': {
                    'strategy': strategy,
                    'hybrid_enabled': self.enable_hybrid,
                    'error': True
                }
            }
    
    def _create_empty_result(self, question: str, strategy: str, processing_time: float) -> Dict:
        """Create result structure for empty search results"""
        return {
            'query': question,
            'answer': 'Không tìm thấy thông tin liên quan đến câu hỏi của bạn.',
            'results': [],
            'context': '',
            'confidence': 0.0,
            'processing_time': processing_time,
            'search_info': {
                'strategy': strategy,
                'hybrid_enabled': self.enable_hybrid,
                'total_results': 0,
                'rerank_enabled': False
            }
        }
    
    def _format_query_results(self, question: str, results: List[HybridRetrievalResult], 
                            strategy: str, processing_time: float,
                            embedding_weight: float, bm25_weight: float, 
                            enable_rerank: bool, include_metadata: bool) -> Dict:
        """Format comprehensive query results"""
        
        # Generate context
        context = self.retriever.format_context(results) if self.enable_hybrid else self._format_standard_context(results)
        
        # Calculate confidence
        confidence = self._calculate_confidence_score(results)
        
        # Format individual results
        formatted_results = self._format_search_results(results, include_metadata)
        
        # Compile retrieval statistics
        retrieval_stats = self._compile_retrieval_stats(results)
        
        return {
            'query': question,
            'results': formatted_results,
            'context': context,
            'confidence': confidence,
            'processing_time': processing_time,
            'search_info': {
                'strategy': strategy,
                'hybrid_enabled': self.enable_hybrid,
                'total_results': len(results),
                'embedding_weight': embedding_weight if self.enable_hybrid else 1.0,
                'bm25_weight': bm25_weight if self.enable_hybrid else 0.0,
                'rerank_enabled': enable_rerank if self.enable_hybrid else False,
                'fusion_method': 'RRF' if strategy == 'hybrid_enhanced' else 'weighted'
            },
            'retrieval_stats': retrieval_stats,
            'answer_context': {
                'total_chunks': len(results),
                'unique_documents': len(set(r.chunk.metadata.get('document_id', 'unknown') for r in results)),
                'content_types': list(set(r.chunk.metadata.get('content_type', 'text') for r in results))
            }
        }
    
    def _format_search_results(self, results: List[HybridRetrievalResult], include_metadata: bool) -> List[Dict]:
        """Format search results with comprehensive information and Excel metadata"""
        formatted_results = []
        
        for i, result in enumerate(results):
            chunk = result.chunk
            metadata = chunk.metadata or {}
            
            # Basic result information
            result_info = {
                'rank': i + 1,
                'score': result.score,
                'chunk_id': result.chunk_id,
                'content': chunk.content,
                'content_preview': chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                
                # Document information from Excel metadata
                'document_info': {
                    'title': metadata.get('document_title', metadata.get('doc_title', 'Không rõ')),
                    'file_name': metadata.get('filename', metadata.get('file_name', 'Không rõ')),
                    'document_id': metadata.get('doc_id', metadata.get('document_id', 'Không rõ')),
                    'topic': metadata.get('topic', 'Không rõ'),
                    'doc_date': metadata.get('doc_date', 'Không rõ')
                },
                
                # Content structure
                'structure_info': {
                    'section_title': metadata.get('section_title', ''),
                    'hierarchy_path': metadata.get('hierarchy_path', ''),
                    'chunk_type': chunk.chunk_type,
                    'section_level': metadata.get('section_level', 0)
                }
            }
            
            # Add hybrid scoring information if available
            if self.enable_hybrid and hasattr(result, 'embedding_score'):
                result_info['scoring'] = {
                    'final_score': result.score,
                    'embedding_score': getattr(result, 'embedding_score', 0.0),
                    'bm25_score': getattr(result, 'bm25_score', 0.0),
                    'rerank_score': getattr(result, 'rerank_score', None),
                    'retrieval_source': getattr(result, 'retrieval_source', 'unknown'),
                    'embedding_rank': result.context_metadata.get('embedding_rank') if hasattr(result, 'context_metadata') else None,
                    'bm25_rank': result.context_metadata.get('bm25_rank') if hasattr(result, 'context_metadata') else None
                }
            
            # Add contextual information
            if hasattr(result, 'parent_chunk') and result.parent_chunk:
                result_info['parent_context'] = {
                    'title': result.parent_chunk.metadata.get('section_title', ''),
                    'content_preview': result.parent_chunk.content[:200] + "..." if len(result.parent_chunk.content) > 200 else result.parent_chunk.content
                }
            
            if hasattr(result, 'related_chunks') and result.related_chunks:
                result_info['related_content'] = [
                    {
                        'title': related.metadata.get('section_title', f'Phần liên quan {j+1}'),
                        'content_preview': related.content[:150] + "..." if len(related.content) > 150 else related.content
                    }
                    for j, related in enumerate(result.related_chunks[:2])
                ]
            
            # Include detailed metadata if requested
            if include_metadata:
                result_info['full_metadata'] = metadata
                if hasattr(result, 'context_metadata'):
                    result_info['retrieval_metadata'] = result.context_metadata
            
            formatted_results.append(result_info)
        
        return formatted_results
    
    def _compile_retrieval_stats(self, results: List[HybridRetrievalResult]) -> Dict:
        """Compile retrieval statistics from results"""
        if not self.enable_hybrid or not results:
            return {}
        
        stats = {
            'embedding_only': 0,
            'bm25_only': 0,
            'hybrid': 0,
            'reranked': 0,
            'avg_embedding_score': 0.0,
            'avg_bm25_score': 0.0,
            'score_distribution': []
        }
        
        embedding_scores = []
        bm25_scores = []
        
        for result in results:
            if hasattr(result, 'retrieval_source'):
                if result.retrieval_source == 'embedding':
                    stats['embedding_only'] += 1
                elif result.retrieval_source == 'bm25':
                    stats['bm25_only'] += 1
                elif result.retrieval_source == 'hybrid':
                    stats['hybrid'] += 1
            
            if hasattr(result, 'rerank_score') and result.rerank_score is not None:
                stats['reranked'] += 1
            
            if hasattr(result, 'embedding_score'):
                embedding_scores.append(result.embedding_score)
            if hasattr(result, 'bm25_score'):
                bm25_scores.append(result.bm25_score)
            
            stats['score_distribution'].append({
                'rank': len(stats['score_distribution']) + 1,
                'final_score': result.score
            })
        
        if embedding_scores:
            stats['avg_embedding_score'] = sum(embedding_scores) / len(embedding_scores)
        if bm25_scores:
            stats['avg_bm25_score'] = sum(bm25_scores) / len(bm25_scores)
        
        return stats
    
    def _calculate_confidence_score(self, results: List[HybridRetrievalResult]) -> float:
        """Calculate confidence score based on result quality"""
        if not results:
            return 0.0
        
        # Weight the top results more heavily
        weights = [0.4, 0.3, 0.2, 0.1]
        weighted_score = 0.0
        total_weight = 0.0
        
        for i, result in enumerate(results[:4]):
            weight = weights[i] if i < len(weights) else 0.05
            
            # Boost score for hybrid retrieval features
            score_multiplier = 1.0
            if hasattr(result, 'retrieval_source') and result.retrieval_source == 'hybrid':
                score_multiplier += 0.15
            if hasattr(result, 'rerank_score') and result.rerank_score is not None:
                score_multiplier += 0.1
            if result.parent_chunk:
                score_multiplier += 0.1
            if result.related_chunks:
                score_multiplier += 0.05
            
            adjusted_score = min(result.score * score_multiplier, 1.0)
            weighted_score += adjusted_score * weight
            total_weight += weight
        
        return min(weighted_score / total_weight if total_weight > 0 else 0.0, 1.0)
    
    def _format_standard_context(self, results) -> str:
        """Fallback context formatting for standard retrieval"""
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[Nguồn {i+1}] {result.chunk.content}")
        return "\n\n".join(context_parts)
    
    def _update_query_stats(self, question: str, processing_time: float, success: bool):
        """Update query statistics"""
        self.query_stats['total_queries'] += 1
        if success:
            self.query_stats['successful_queries'] += 1
        else:
            self.query_stats['failed_queries'] += 1
        
        # Update average response time
        total_time = (self.query_stats['average_response_time'] * 
                     (self.query_stats['total_queries'] - 1) + processing_time)
        self.query_stats['average_response_time'] = total_time / self.query_stats['total_queries']
        
        # Keep history of recent queries (last 10)
        self.query_stats['query_history'].append({
            'query': question[:100] + "..." if len(question) > 100 else question,
            'processing_time': processing_time,
            'success': success,
            'timestamp': time.time()
        })
        
        if len(self.query_stats['query_history']) > 10:
            self.query_stats['query_history'].pop(0)
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        vector_stats = self.vector_store.get_stats()
        
        stats = {
            'query_statistics': self.query_stats,
            'vector_store_stats': vector_stats,
            'index_path': self.index_path,
            'system_config': {
                'hybrid_enabled': self.enable_hybrid,
                'embedding_model': self.config.embedding.model_name,
                'vector_store_type': self.config.vector_store.store_type
            }
        }
        
        # Add hybrid-specific stats
        if self.enable_hybrid and hasattr(self.retriever, 'get_stats'):
            hybrid_stats = self.retriever.get_stats()
            stats['hybrid_retrieval_stats'] = hybrid_stats
        
        # Load index creation stats if available
        try:
            stats_path = f"{self.index_path}.stats"
            if os.path.exists(stats_path):
                with open(stats_path, 'r', encoding='utf-8') as f:
                    index_stats = json.load(f)
                    stats['index_creation_stats'] = index_stats
        except Exception as e:
            logger.debug(f"Could not load index stats: {e}")
        
        return stats
    
    def batch_query(self, questions: List[str], **query_params) -> List[Dict]:
        """Execute multiple queries in batch"""
        logger.info(f"🔍 Processing batch of {len(questions)} queries")
        
        results = []
        for i, question in enumerate(questions):
            logger.info(f"  Query {i+1}/{len(questions)}: {question[:50]}...")
            result = self.query(question, **query_params)
            results.append(result)
        
        logger.info(f"✅ Batch query completed: {len(results)} results")
        return results

def display_search_results(query_result: Dict, show_details: bool = True):
    """Display search results in a formatted way with Excel metadata"""
    print("\n" + "="*120)
    print(f"🔍 SEARCH RESULTS")
    print("="*120)
    
    print(f"📝 Query: {query_result['query']}")
    print(f"⏱️ Processing time: {query_result['processing_time']:.3f}s")
    print(f"🎯 Confidence: {query_result['confidence']:.3f}")
    
    search_info = query_result.get('search_info', {})
    print(f"🔧 Strategy: {search_info.get('strategy', 'unknown')}")
    print(f"🔄 Hybrid: {search_info.get('hybrid_enabled', False)}")
    
    if search_info.get('hybrid_enabled'):
        print(f"⚖️ Weights: Embedding {search_info.get('embedding_weight', 0.7):.1f} - BM25 {search_info.get('bm25_weight', 0.3):.1f}")
        print(f"🎯 Rerank: {search_info.get('rerank_enabled', False)}")
    
    # Display retrieval statistics
    retrieval_stats = query_result.get('retrieval_stats', {})
    if retrieval_stats:
        print(f"\n📊 Retrieval Statistics:")
        if retrieval_stats.get('hybrid', 0) > 0:
            print(f"   • Hybrid matches: {retrieval_stats['hybrid']}")
        if retrieval_stats.get('embedding_only', 0) > 0:
            print(f"   • Embedding only: {retrieval_stats['embedding_only']}")
        if retrieval_stats.get('bm25_only', 0) > 0:
            print(f"   • BM25 only: {retrieval_stats['bm25_only']}")
        if retrieval_stats.get('reranked', 0) > 0:
            print(f"   • Reranked: {retrieval_stats['reranked']}")
    
    # Display answer context
    answer_context = query_result.get('answer_context', {})
    if answer_context:
        print(f"\n📋 Answer Context:")
        print(f"   • Total chunks: {answer_context.get('total_chunks', 0)}")
        print(f"   • Unique documents: {answer_context.get('unique_documents', 0)}")
        content_types = answer_context.get('content_types', [])
        if content_types:
            print(f"   • Content types: {', '.join(content_types)}")
    
    results = query_result.get('results', [])
    if not results:
        print(f"\n❌ No results found")
        return
    
    print(f"\n📄 Found {len(results)} results:")
    
    for result in results:
        print(f"\n{'='*120}")
        print(f"📄 **RESULT #{result['rank']}**")
        
        # Scoring information
        if 'scoring' in result and show_details:
            scoring = result['scoring']
            score_parts = [f"Final: {scoring['final_score']:.4f}"]
            
            if scoring.get('embedding_score', 0) > 0:
                score_parts.append(f"Embedding: {scoring['embedding_score']:.3f}")
            if scoring.get('bm25_score', 0) > 0:
                score_parts.append(f"BM25: {scoring['bm25_score']:.3f}")
            if scoring.get('rerank_score') is not None:
                score_parts.append(f"Rerank: {scoring['rerank_score']:.3f}")
            
            print(f"🎯 Scores: {' | '.join(score_parts)}")
            print(f"📊 Source: {scoring.get('retrieval_source', 'unknown')}")
            
            # Ranking information
            ranks = []
            if scoring.get('embedding_rank') and scoring['embedding_rank'] < 999:
                ranks.append(f"Embedding #{scoring['embedding_rank']}")
            if scoring.get('bm25_rank') and scoring['bm25_rank'] < 999:
                ranks.append(f"BM25 #{scoring['bm25_rank']}")
            if ranks:
                print(f"🏆 Rankings: {' + '.join(ranks)}")
        else:
            print(f"🎯 Score: {result['score']:.4f}")
        
        # Document information with Excel metadata
        doc_info = result.get('document_info', {})
        print(f"📋 Document: {doc_info.get('title', 'Không rõ')}")
        print(f"📁 File: {doc_info.get('file_name', 'Không rõ')}")
        
        # Safely display doc_id from Excel metadata
        doc_id = doc_info.get('doc_id', 'Không rõ')
        print(f"🆔 Doc ID: {doc_id}")
        
        # Show additional Excel metadata
        topic = doc_info.get('topic', 'Không rõ')
        if topic != 'Không rõ':
            print(f"🏷️ Topic: {topic}")
            
        doc_date = doc_info.get('doc_date', 'Không rõ')
        if doc_date != 'Không rõ':
            print(f"📅 Date: {doc_info['doc_date']}")
        
        # Structure information
        struct_info = result['structure_info']
        if struct_info['hierarchy_path']:
            print(f"📍 Path: {struct_info['hierarchy_path']}")
        if struct_info['section_title']:
            print(f"📑 Section: {struct_info['section_title']}")
        
        print(f"📄 Type: {struct_info['chunk_type']}")
        
        # Main content
        print(f"\n📄 **CONTENT:**")
        print("-" * 100)
        print(result['content_preview'])
        
        # Parent context
        if 'parent_context' in result and show_details:
            parent = result['parent_context']
            print(f"\n👆 **PARENT CONTEXT:**")
            print(f"   📌 {parent['title']}")
            print("   " + "-" * 80)
            for line in parent['content_preview'].split('\n')[:3]:
                print(f"   {line}")
        
        # Related content
        if 'related_content' in result and show_details:
            related_content = result['related_content']
            if related_content:
                print(f"\n🔗 **RELATED CONTENT:**")
                for related in related_content:
                    print(f"   • {related['title']}")
                    print(f"     {related['content_preview'][:100]}...")
        
        # Show full metadata if requested and available
        if show_details and 'full_metadata' in result:
            metadata = result['full_metadata']
            excel_fields = ['doc_id', 'doc_title', 'filename', 'doc_date', 'topic']
            excel_metadata = {k: v for k, v in metadata.items() if k in excel_fields and v}
            
            if excel_metadata:
                print(f"\n📊 **EXCEL METADATA:**")
                for key, value in excel_metadata.items():
                    print(f"   • {key}: {value}")
        
        print("\n" + "-"*120)


def create_query_result_summary(query_result: Dict) -> Dict:
    """Create a comprehensive summary of query results with Excel metadata"""
    summary = {
        'query': query_result['query'],
        'total_results': len(query_result.get('results', [])),
        'confidence': query_result['confidence'],
        'processing_time': query_result['processing_time'],
        'search_strategy': query_result.get('search_info', {}).get('strategy', 'unknown'),
        'hybrid_enabled': query_result.get('search_info', {}).get('hybrid_enabled', False),
        'documents_found': set(),
        'topics_found': set(),
        'content_types_found': set(),
        'date_range': {'earliest': None, 'latest': None},
        'top_scoring_result': None
    }
    
    results = query_result.get('results', [])
    if not results:
        return summary
    
    # Analyze results
    dates = []
    scores = []
    
    for result in results:
        doc_info = result.get('document_info', {})
        
        # Collect document IDs and topics
        if doc_info.get('doc_id'):
            summary['documents_found'].add(doc_info['doc_id'])
        elif doc_info.get('document_id'):
            summary['documents_found'].add(doc_info['document_id'])
        if doc_info.get('topic'):
            summary['topics_found'].add(doc_info['topic'])
        
        # Collect content types
        struct_info = result.get('structure_info', {})
        if struct_info.get('chunk_type'):
            summary['content_types_found'].add(struct_info['chunk_type'])
        
        # Collect dates
        if doc_info.get('doc_date'):
            try:
                # Handle different date formats
                date_str = doc_info['doc_date']
                if isinstance(date_str, str) and date_str.strip():
                    dates.append(date_str)
            except:
                pass
        
        # Collect scores
        scores.append(result.get('score', 0))
    
    # Convert sets to lists and get counts
    summary['documents_found'] = list(summary['documents_found'])
    summary['topics_found'] = list(summary['topics_found'])
    summary['content_types_found'] = list(summary['content_types_found'])
    summary['unique_documents_count'] = len(summary['documents_found'])
    summary['topics_count'] = len(summary['topics_found'])
    
    # Date range analysis
    if dates:
        summary['date_range']['earliest'] = min(dates)
        summary['date_range']['latest'] = max(dates)
    
    # Top scoring result
    if results:
        summary['top_scoring_result'] = {
            'rank': 1,
            'score': results[0].get('score', 0),
            'document_title': results[0].get('document_info', {}).get('title', 'Unknown'),
            'doc_id': results[0].get('document_info', {}).get('doc_id', 'Unknown'),
            'topic': results[0].get('document_info', {}).get('topic', 'Unknown'),
            'content_preview': results[0].get('content_preview', '')[:100] + "..."
        }
    
    # Score statistics
    if scores:
        summary['score_stats'] = {
            'max_score': max(scores),
            'min_score': min(scores),
            'avg_score': sum(scores) / len(scores),
            'score_variance': sum((s - summary['score_stats']['avg_score'])**2 for s in scores) / len(scores)
        }
    
    return summary


def save_query_results(query_result: Dict, output_file: str):
    """Save query results to file with Excel metadata preserved"""
    try:
        # Create a comprehensive save structure
        save_data = {
            'query_info': {
                'query': query_result['query'],
                'timestamp': time.time(),
                'processing_time': query_result['processing_time'],
                'confidence': query_result['confidence']
            },
            'search_configuration': query_result.get('search_info', {}),
            'retrieval_statistics': query_result.get('retrieval_stats', {}),
            'answer_context': query_result.get('answer_context', {}),
            'results': []
        }
        
        # Process and save each result with Excel metadata
        for result in query_result.get('results', []):
            result_data = {
                'rank': result['rank'],
                'score': result['score'],
                'content': result['content'],
                'content_preview': result['content_preview'],
                'document_metadata': result['document_info'],  # Includes Excel metadata
                'structure_info': result['structure_info'],
                'scoring_details': result.get('scoring', {}),
                'parent_context': result.get('parent_context'),
                'related_content': result.get('related_content', [])
            }
            
            # Include full metadata if available
            if 'full_metadata' in result:
                result_data['full_metadata'] = result['full_metadata']
            
            save_data['results'].append(result_data)
        
        # Add summary
        save_data['summary'] = create_query_result_summary(query_result)
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Query results saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving query results: {e}")
        return False

def display_search_results_with_doc_id(query_result: Dict, show_details: bool = True):
    """Enhanced display function with prominent doc_id display"""
    print("\n" + "="*120)
    print(f"🔍 SEARCH RESULTS WITH DOC_ID")
    print("="*120)
    
    print(f"📝 Query: {query_result['query']}")
    print(f"⏱️ Processing time: {query_result['processing_time']:.3f}s")
    print(f"🎯 Confidence: {query_result['confidence']:.3f}")
    
    search_info = query_result.get('search_info', {})
    print(f"🔧 Strategy: {search_info.get('strategy', 'unknown')}")
    print(f"🔄 Hybrid: {search_info.get('hybrid_enabled', False)}")
    
    if search_info.get('hybrid_enabled'):
        print(f"⚖️ Weights: Embedding {search_info.get('embedding_weight', 0.7):.1f} - BM25 {search_info.get('bm25_weight', 0.3):.1f}")
        print(f"🎯 Rerank: {search_info.get('rerank_enabled', False)}")
    
    # Display answer context
    answer_context = query_result.get('answer_context', {})
    if answer_context:
        print(f"\n📋 Answer Context:")
        print(f"   • Total chunks: {answer_context.get('total_chunks', 0)}")
        print(f"   • Unique documents: {answer_context.get('unique_documents', 0)}")
        content_types = answer_context.get('content_types', [])
        if content_types:
            print(f"   • Content types: {', '.join(content_types)}")
    
    results = query_result.get('results', [])
    if not results:
        print(f"\n❌ No results found")
        return
    
    print(f"\n📄 Found {len(results)} results:")
    
    # Show document summary first
    doc_summary = {}
    for result in results:
        doc_info = result.get('document_info', {})
        doc_id = doc_info.get('doc_id', 'Unknown')
        topic = doc_info.get('topic', 'Unknown')
        title = doc_info.get('title', 'Unknown')
        
        if doc_id not in doc_summary:
            doc_summary[doc_id] = {
                'title': title,
                'topic': topic,
                'chunk_count': 0
            }
        doc_summary[doc_id]['chunk_count'] += 1
    
    print(f"\n📚 Documents found:")
    for doc_id, info in doc_summary.items():
        print(f"  🆔 {doc_id}: {info['title']} ({info['topic']}) - {info['chunk_count']} chunks")
    
    # Show detailed results
    for result in results:
        print(f"\n{'='*120}")
        print(f"📄 **RESULT #{result['rank']}**")
        
        # Scoring information
        if 'scoring' in result and show_details:
            scoring = result['scoring']
            score_parts = [f"Final: {scoring['final_score']:.4f}"]
            
            if scoring.get('embedding_score', 0) > 0:
                score_parts.append(f"Embedding: {scoring['embedding_score']:.3f}")
            if scoring.get('bm25_score', 0) > 0:
                score_parts.append(f"BM25: {scoring['bm25_score']:.3f}")
            if scoring.get('rerank_score') is not None:
                score_parts.append(f"Rerank: {scoring['rerank_score']:.3f}")
            
            print(f"🎯 Scores: {' | '.join(score_parts)}")
            print(f"📊 Source: {scoring.get('retrieval_source', 'unknown')}")
        else:
            print(f"🎯 Score: {result['score']:.4f}")
        
        # Document information with prominent doc_id
        doc_info = result['document_info']
        print(f"🆔 DOC_ID: {doc_info['doc_id']}")
        print(f"📋 Document: {doc_info['title']}")
        print(f"📁 File: {doc_info['file_name']}")
        print(f"🏷️ Topic: {doc_info['topic']}")
        print(f"📅 Date: {doc_info['doc_date']}")
        
        # Structure information
        struct_info = result['structure_info']
        if struct_info['hierarchy_path']:
            print(f"📍 Path: {struct_info['hierarchy_path']}")
        if struct_info['section_title']:
            print(f"📑 Section: {struct_info['section_title']}")
        
        print(f"📄 Type: {struct_info['chunk_type']}")
        
        # Main content
        print(f"\n📄 **CONTENT:**")
        print("-" * 100)
        print(result['content_preview'])
        
        print("\n" + "-"*120)

def interactive_search_session(query_system: DocumentQuerySystem):
    """Enhanced interactive search session with doc_id display"""
    print("\n" + "="*80)
    print("🔍 ENHANCED INTERACTIVE DOCUMENT SEARCH")
    print("="*80)
    print("Commands:")
    print("  - Type your question to search")
    print("  - 'stats' to show system statistics")
    print("  - 'strategy [name]' to change search strategy")
    print("    Available: hybrid_enhanced, embedding_only, bm25_only, ensemble")
    print("  - 'weights [emb] [bm25]' to adjust fusion weights")
    print("  - 'rerank [on/off]' to toggle reranking")
    print("  - 'results [num]' to change number of results (default: 5)")
    print("  - 'details [on/off]' to toggle detailed output")
    print("  - 'save [filename]' to save last query results")
    print("  - 'docs' to show available document info")
    print("  - 'quit' to exit")
    print("="*80)
    
    # Default settings
    current_strategy = 'hybrid_enhanced'
    embedding_weight = 0.7
    bm25_weight = 0.3
    enable_rerank = True
    num_results = 5
    show_details = True
    last_result = None
    
    print(f"📋 Settings: {current_strategy} | Weights: {embedding_weight:.1f}/{bm25_weight:.1f} | Results: {num_results} | Rerank: {enable_rerank}")
    
    while True:
        try:
            user_input = input(f"\n❓ Search [{current_strategy}]: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                stats = query_system.get_system_stats()
                print("\n📊 System Statistics:")
                
                # Query stats
                query_stats = stats.get('query_statistics', {})
                print(f"  🔍 Queries processed: {query_stats.get('total_queries', 0)}")
                print(f"  ✅ Successful: {query_stats.get('successful_queries', 0)}")
                print(f"  ❌ Failed: {query_stats.get('failed_queries', 0)}")
                print(f"  ⏱️ Avg response time: {query_stats.get('average_response_time', 0):.3f}s")
                
                # Vector store stats
                vs_stats = stats.get('vector_store_stats', {})
                print(f"  💾 Vector store chunks: {vs_stats.get('active_chunks', 0)}")
                
                # Index creation stats
                index_stats = stats.get('index_creation_stats', {})
                if index_stats:
                    print(f"  📚 Documents indexed: {index_stats.get('documents_processed', 0)}")
                    print(f"  🏷️ Topics processed: {len(index_stats.get('topics_processed', {}))}")
                    
                    for topic, topic_data in index_stats.get('topics_processed', {}).items():
                        print(f"    • {topic}: {topic_data.get('documents', 0)} docs")
                
            elif user_input.lower() == 'docs':
                # Show available document information
                stats = query_system.get_system_stats()
                index_stats = stats.get('index_creation_stats', {})
                
                if index_stats and 'successful_documents' in index_stats:
                    docs = index_stats['successful_documents']
                    print(f"\n📚 Available Documents ({len(docs)} total):")
                    
                    # Group by topic
                    by_topic = {}
                    for doc in docs:
                        topic = doc.get('topic', 'Unknown')
                        if topic not in by_topic:
                            by_topic[topic] = []
                        by_topic[topic].append(doc)
                    
                    for topic, topic_docs in by_topic.items():
                        print(f"\n🏷️ {topic} ({len(topic_docs)} documents):")
                        for doc in topic_docs[:5]:  # Show first 5
                            print(f"  📄 {doc.get('document_id', 'Unknown')} - {doc.get('filename', 'Unknown')}")
                            print(f"      {doc.get('chunks_count', 0)} chunks")
                        
                        if len(topic_docs) > 5:
                            print(f"  ... and {len(topic_docs) - 5} more")
                else:
                    print("📚 No document information available")
                    
            elif user_input.lower().startswith('save '):
                if last_result:
                    filename = user_input.split(' ', 1)[1].strip()
                    if not filename.endswith('.json'):
                        filename += '.json'
                    
                    success = save_query_results(last_result, filename)
                    if success:
                        print(f"✅ Results saved to {filename}")
                    else:
                        print(f"❌ Failed to save results")
                else:
                    print("❌ No results to save - execute a query first")
                    
            elif user_input.lower().startswith('strategy '):
                new_strategy = user_input.split(' ', 1)[1].strip()
                valid_strategies = ['hybrid_enhanced', 'embedding_only', 'bm25_only', 'ensemble']
                if new_strategy in valid_strategies:
                    current_strategy = new_strategy
                    print(f"✅ Strategy: {current_strategy}")
                else:
                    print(f"❌ Invalid strategy. Available: {', '.join(valid_strategies)}")
                    
            elif user_input.lower().startswith('weights '):
                try:
                    parts = user_input.split()
                    if len(parts) == 3:
                        new_emb = float(parts[1])
                        new_bm25 = float(parts[2])
                        if 0 <= new_emb <= 1 and 0 <= new_bm25 <= 1:
                            embedding_weight = new_emb
                            bm25_weight = new_bm25
                            print(f"✅ Weights: Embedding {embedding_weight:.1f}, BM25 {bm25_weight:.1f}")
                        else:
                            print("❌ Weights must be between 0 and 1")
                    else:
                        print("❌ Usage: weights [embedding_weight] [bm25_weight]")
                except ValueError:
                    print("❌ Invalid weight values")
                    
            elif user_input.lower().startswith('rerank '):
                setting = user_input.split(' ', 1)[1].strip().lower()
                if setting in ['on', 'true', 'yes']:
                    enable_rerank = True
                    print("✅ Reranking enabled")
                elif setting in ['off', 'false', 'no']:
                    enable_rerank = False
                    print("✅ Reranking disabled")
                else:
                    print("❌ Usage: rerank [on/off]")
                    
            elif user_input.lower().startswith('results '):
                try:
                    new_num = int(user_input.split(' ', 1)[1].strip())
                    if 1 <= new_num <= 20:
                        num_results = new_num
                        print(f"✅ Number of results: {num_results}")
                    else:
                        print("❌ Number of results must be between 1 and 20")
                except ValueError:
                    print("❌ Invalid number")
                    
            elif user_input.lower().startswith('details '):
                setting = user_input.split(' ', 1)[1].strip().lower()
                if setting in ['on', 'true', 'yes']:
                    show_details = True
                    print("✅ Detailed output enabled")
                elif setting in ['off', 'false', 'no']:
                    show_details = False
                    print("✅ Detailed output disabled")
                else:
                    print("❌ Usage: details [on/off]")
                    
            elif user_input:
                # Execute search
                result = query_system.query(
                    question=user_input,
                    k=num_results,
                    strategy=current_strategy,
                    embedding_weight=embedding_weight,
                    bm25_weight=bm25_weight,
                    enable_rerank=enable_rerank,
                    include_metadata=True
                )
                
                last_result = result  # Store for potential saving
                
                # Enhanced result display with doc_id
                display_search_results_with_doc_id(result, show_details)
                
                # Quick summary
                results = result.get('results', [])
                if results:
                    unique_docs = set()
                    topics = set()
                    for r in results:
                        doc_info = r.get('document_info', {})
                        if doc_info.get('doc_id'):
                            unique_docs.add(doc_info['doc_id'])
                        if doc_info.get('topic'):
                            topics.add(doc_info['topic'])
                    
                    print(f"\n📊 Quick Summary:")
                    print(f"   📄 Unique documents: {len(unique_docs)}")
                    if topics:
                        print(f"   🏷️ Topics: {', '.join(topics)}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Enhanced search session ended!")

async def main():
    """Main function for document querying"""
    parser = argparse.ArgumentParser(description='Vietnamese Document Query System')
    parser.add_argument('--index-path', default='storage/index',
                       help='Path to load vector index from')
    parser.add_argument('--query', help='Single query to execute')
    parser.add_argument('--batch-queries', help='File containing queries (one per line)')
    parser.add_argument('--strategy', default='hybrid_enhanced',
                       choices=['hybrid_enhanced', 'embedding_only', 'bm25_only', 'ensemble'],
                       help='Search strategy')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of results to return')
    parser.add_argument('--embedding-weight', type=float, default=0.7,
                       help='Weight for embedding scores')
    parser.add_argument('--bm25-weight', type=float, default=0.3,
                       help='Weight for BM25 scores')
    parser.add_argument('--no-rerank', action='store_true',
                       help='Disable reranking')
    parser.add_argument('--disable-hybrid', action='store_true',
                       help='Disable hybrid search')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive search session')
    parser.add_argument('--stats', action='store_true',
                       help='Show system statistics')
    parser.add_argument('--no-details', action='store_true',
                       help='Hide detailed output')
    parser.add_argument('--config-preset', default='development',
                       choices=['development', 'production', 'high_memory', 'low_memory'],
                       help='Configuration preset')
    
    args = parser.parse_args()
    
    # Initialize configuration
    print("🚀 Initializing Document Query System...")
    
    config_presets_map = {
        'development': ConfigPresets.development(),
        'production': ConfigPresets.production(),
        'high_memory': ConfigPresets.high_memory(),
        'low_memory': ConfigPresets.low_memory()
    }
    
    config = config_presets_map[args.config_preset]
    print(f"📋 Config: {args.config_preset}")
    print(f"🔄 Hybrid search: {not args.disable_hybrid}")
    
    # Initialize query system
    query_system = DocumentQuerySystem(
        config=config,
        index_path=args.index_path,
        enable_hybrid=not args.disable_hybrid
    )
    
    # Check if index is loaded
    stats = query_system.get_system_stats()
    vector_chunks = stats['vector_store_stats'].get('active_chunks', 0)
    
    if vector_chunks == 0:
        print("❌ No documents found in index!")
        print("💡 Please run indexing.py first to create the index")
        return
    
    print(f"✅ Loaded index with {vector_chunks} chunks")
    
    # Show system statistics if requested
    if args.stats:
        print("\n📊 System Statistics:")
        print("="*50)
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        return
    
    # Execute single query
    if args.query:
        print(f"🔍 Executing query: {args.query}")
        result = query_system.query(
            question=args.query,
            k=args.k,
            strategy=args.strategy,
            embedding_weight=args.embedding_weight,
            bm25_weight=args.bm25_weight,
            enable_rerank=not args.no_rerank
        )
        display_search_results(result, not args.no_details)
        return
    
    # Execute batch queries
    if args.batch_queries:
        if not os.path.exists(args.batch_queries):
            print(f"❌ Batch queries file not found: {args.batch_queries}")
            return
        
        with open(args.batch_queries, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"🔍 Executing {len(queries)} batch queries...")
        results = query_system.batch_query(
            queries,
            k=args.k,
            strategy=args.strategy,
            embedding_weight=args.embedding_weight,
            bm25_weight=args.bm25_weight,
            enable_rerank=not args.no_rerank
        )
        
        for i, result in enumerate(results):
            print(f"\n{'='*120}")
            print(f"BATCH QUERY {i+1}/{len(queries)}")
            print('='*120)
            display_search_results(result, not args.no_details)
        
        return
    
    # Start interactive session (default)
    if args.interactive or not any([args.query, args.batch_queries, args.stats]):
        interactive_search_session(query_system)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())