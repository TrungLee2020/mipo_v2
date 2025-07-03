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
import datetime
import pandas as pd

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
            include_metadata: bool = False) -> Dict:  # Changed default to False
        """Execute search query with simplified metadata by default"""
        
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing query: '{question[:50]}...'")
            
            # Execute retrieval
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
                results = self.retriever.retrieve(
                    query=question,
                    k=k,
                    retrieval_strategy='enhanced_hierarchical'
                )
            
            processing_time = time.time() - start_time
            
            if not results:
                return self._create_empty_result(question, strategy, processing_time)
            
            # Format results with simplified structure
            formatted_results = self._format_search_results(results, include_metadata)
            
            # Calculate confidence
            confidence = self._calculate_confidence_score(results)
            
            # Create simplified response
            query_result = {
                'query': question,
                'results': formatted_results,
                'confidence': confidence,
                'processing_time': processing_time,
                'total_results': len(results),
            }
            
            # Add search info only if requested
            if include_metadata:
                query_result['search_info'] = {
                    'strategy': strategy,
                    'hybrid_enabled': self.enable_hybrid,
                    'embedding_weight': embedding_weight if self.enable_hybrid else 1.0,
                    'bm25_weight': bm25_weight if self.enable_hybrid else 0.0,
                    'rerank_enabled': enable_rerank if self.enable_hybrid else False,
                }
            
            self._update_query_stats(question, processing_time, True)
            logger.info(f"✅ Query completed in {processing_time:.3f}s - Found {len(results)} results")
            
            return query_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Query error: {e}")
            self._update_query_stats(question, processing_time, False)
            
            return {
                'query': question,
                'results': [],
                'confidence': 0.0,
                'processing_time': processing_time,
                'total_results': 0,
                'error': str(e)
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

    def _format_doc_date(self, doc_date_value) -> str:
        """Format doc_date from various formats to string"""
        if not doc_date_value:
            return 'Unknown'
        
        # If already a string, return as is
        if isinstance(doc_date_value, str):
            return doc_date_value
        
        # Handle pandas Timestamp
        try:
            if isinstance(doc_date_value, pd.Timestamp):
                return doc_date_value.strftime('%Y-%m-%d')
        except ImportError:
            pass
        
        # Handle datetime objects
        try:
            import datetime
            if isinstance(doc_date_value, (datetime.datetime, datetime.date)):
                return doc_date_value.strftime('%Y-%m-%d')
        except:
            pass
        
        # Handle numpy datetime
        try:
            import numpy as np
            if isinstance(doc_date_value, np.datetime64):
                return str(doc_date_value)[:10]  # Get YYYY-MM-DD part
        except:
            pass
        
        # Fallback: convert to string
        try:
            return str(doc_date_value)
        except:
            return 'Unknown'

    def _format_search_results(self, results: List[HybridRetrievalResult], include_metadata: bool) -> List[Dict]:
        """Format search results with simplified structure and proper doc_id extraction"""
        formatted_results = []
        
        for i, result in enumerate(results):
            chunk = result.chunk
            metadata = chunk.metadata or {}
            
            # Extract doc_id properly from metadata (prioritize doc_id over document_id)
            doc_id = metadata.get('doc_id')
            
            # Format doc_date properly (handle different formats)
            doc_date = self._format_doc_date(metadata.get('doc_date'))
            
            # Basic result information with simplified structure
            result_info = {
                'rank': i + 1,
                'score': result.score,
                'chunk_id': result.chunk_id,
                'content': chunk.content,
                'content_preview': chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                
                # Simplified document information
                'doc_id': doc_id,
                'document_title': metadata.get('document_title') or metadata.get('doc_title', 'Unknown'),
                'filename': metadata.get('filename') or metadata.get('file_name', 'Unknown'),
                'topic': metadata.get('topic', 'Unknown'),
                'doc_date': doc_date,
                
                # Simplified structure info
                'section_title': metadata.get('section_title', ''),
                'hierarchy_path': metadata.get('hierarchy_path', ''),
                'chunk_type': chunk.chunk_type,
                
                # Parent and child content for better context
                'parent_content': None,
                'child_contents': []
            }
            
            # Add parent content if available
            if result.parent_chunk:
                parent_preview = result.parent_chunk.content[:400] + "..." if len(result.parent_chunk.content) > 400 else result.parent_chunk.content
                result_info['parent_content'] = {
                    'title': result.parent_chunk.metadata.get('section_title', 'Parent Section'),
                    'content': parent_preview,
                    'chunk_type': result.parent_chunk.chunk_type
                }
            
            # Add related/child content if available
            if result.related_chunks:
                for related_chunk in result.related_chunks[:2]:  # Limit to 2 related chunks
                    related_preview = related_chunk.content[:200] + "..." if len(related_chunk.content) > 200 else related_chunk.content
                    result_info['child_contents'].append({
                        'title': related_chunk.metadata.get('section_title', 'Related Section'),
                        'content': related_preview,
                        'chunk_type': related_chunk.chunk_type
                    })
            
            # Add hybrid scoring if available
            if self.enable_hybrid and hasattr(result, 'embedding_score'):
                result_info['scoring'] = {
                    'final_score': result.score,
                    'embedding_score': getattr(result, 'embedding_score', 0.0),
                    'bm25_score': getattr(result, 'bm25_score', 0.0),
                    'retrieval_source': getattr(result, 'retrieval_source', 'unknown')
                }
            
            # Only include full metadata if explicitly requested
            if include_metadata:
                result_info['full_metadata'] = metadata
            
            formatted_results.append(result_info)
        
        return formatted_results

def display_search_results(query_result: Dict, show_details: bool = False):
    """Simplified display function with proper doc_id and parent-child content"""
    print("\n" + "="*80)
    print(f"🔍 SEARCH RESULTS")
    print("="*80)
    
    print(f"📝 Query: {query_result['query']}")
    print(f"⏱️ Time: {query_result['processing_time']:.3f}s")
    print(f"🎯 Confidence: {query_result['confidence']:.2f}")
    
    results = query_result.get('results', [])
    if not results:
        print(f"\n❌ No results found")
        return
    
    print(f"\n📄 Found {len(results)} results:")
    
    for result in results:
        print(f"\n{'='*80}")
        print(f"📄 RESULT #{result['rank']}")
        print(f"🎯 Score: {result['score']:.4f}")
        
        # Display doc_id prominently
        print(f"🆔 DOC_ID: {result['doc_id']}")
        print(f"📋 Title: {result['document_title']}")
        print(f"📁 File: {result['filename']}")
        print(f"🏷️ Topic: {result['topic']}")
        print(f"📅 Date: {result['doc_date']}")
        
        if result['section_title']:
            print(f"📑 Section: {result['section_title']}")
        
        if result['hierarchy_path']:
            print(f"📍 Path: {result['hierarchy_path']}")
        
        print(f"📄 Type: {result['chunk_type']}")
        
        # Show scoring details if available and requested
        if show_details and 'scoring' in result:
            scoring = result['scoring']
            print(f"📊 Scoring: Emb:{scoring['embedding_score']:.3f} | BM25:{scoring['bm25_score']:.3f} | Source:{scoring['retrieval_source']}")
        
        # Show parent content if available
        if result.get('parent_content'):
            parent = result['parent_content']
            print(f"\n👆 PARENT CONTEXT ({parent['chunk_type']}):")
            print(f"📌 {parent['title']}")
            print("-" * 60)
            print(parent['content'])
            print("-" * 60)
        
        # Main content
        print(f"\n📄 MAIN CONTENT ({result['chunk_type']}):")
        print("-" * 60)
        print(result['content_preview'])
        print("-" * 60)
        
        # Show related/child content if available
        if result.get('child_contents'):
            print(f"\n🔗 RELATED CONTENT ({len(result['child_contents'])} items):")
            for i, child in enumerate(result['child_contents']):
                print(f"\n  📌 {i+1}. {child['title']} ({child['chunk_type']}):")
                print(f"     {child['content']}")
        
        print("-" * 80)

def create_simple_query_summary(query_result: Dict) -> Dict:
    """Create simple summary with doc_id information"""
    summary = {
        'query': query_result['query'],
        'total_results': len(query_result.get('results', [])),
        'confidence': query_result['confidence'],
        'processing_time': query_result['processing_time'],
        'documents_found': [],
        'topics_found': set(),
    }
    
    results = query_result.get('results', [])
    if not results:
        return summary
    
    # Collect unique documents and topics
    for result in results:
        doc_info = {
            'doc_id': result['doc_id'],
            'title': result['document_title'], 
            'topic': result['topic'],
            'filename': result['filename']
        }
        
        # Add to documents if not already present
        if not any(d['doc_id'] == doc_info['doc_id'] for d in summary['documents_found']):
            summary['documents_found'].append(doc_info)
        
        summary['topics_found'].add(result['topic'])
    
    # Convert set to list
    summary['topics_found'] = list(summary['topics_found'])
    summary['unique_documents_count'] = len(summary['documents_found'])
    
    return summary

# Add these utility functions at the end of the file

def convert_query_result_to_json(query_result: Dict, pretty_print: bool = True) -> str:
    """Convert query result to JSON string with proper serialization"""
    import json
    import datetime
    import numpy as np
    
    def json_serializer(obj):
        """Custom JSON serializer for complex objects"""
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    try:
        if pretty_print:
            return json.dumps(query_result, ensure_ascii=False, indent=2, default=json_serializer)
        else:
            return json.dumps(query_result, ensure_ascii=False, default=json_serializer)
    except Exception as e:
        logger.error(f"Error converting to JSON: {e}")
        return json.dumps({'error': f'JSON conversion failed: {str(e)}'}, ensure_ascii=False)

def save_query_results_json(query_result: Dict, output_file: str, include_summary: bool = True) -> bool:
    """Save query results to JSON file with optional summary"""
    try:
        # Create output structure
        output_data = {
            'query_info': {
                'query': query_result['query'],
                'timestamp': datetime.now().isoformat(),
                'processing_time': query_result['processing_time'],
                'confidence': query_result['confidence'],
                'total_results': query_result.get('total_results', len(query_result.get('results', [])))
            },
            'results': query_result.get('results', [])
        }
        
        # Add summary if requested
        if include_summary:
            output_data['summary'] = create_simple_query_summary(query_result)
        
        # Add search configuration if available
        if 'search_info' in query_result:
            output_data['search_config'] = query_result['search_info']
        
        # Write to file
        json_content = convert_query_result_to_json(output_data, pretty_print=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_content)
        
        print(f"💾 Query results saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving query results: {e}")
        return False

def interactive_search_session_simplified(query_system: DocumentQuerySystem):
    """Simplified interactive search session with JSON export"""
    print("\n" + "="*60)
    print("🔍 DOCUMENT SEARCH")
    print("="*60)
    print("Commands:")
    print("  - Type your question to search")
    print("  - 'stats' for system statistics")
    print("  - 'docs' to show available documents")
    print("  - 'details [on/off]' to toggle detailed output")
    print("  - 'save [filename]' to save last result as JSON")
    print("  - 'json' to show last result as JSON")
    print("  - 'quit' to exit")
    print("="*60)
    
    show_details = False
    last_result = None
    
    while True:
        try:
            user_input = input(f"\n❓ Search: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'stats':
                stats = query_system.get_system_stats()
                print(f"\n📊 System Stats:")
                print(f"   Documents indexed: {stats.get('index_creation_stats', {}).get('documents_processed', 0)}")
                print(f"   Vector store chunks: {stats.get('vector_store_stats', {}).get('active_chunks', 0)}")
                print(f"   Queries processed: {stats.get('query_statistics', {}).get('total_queries', 0)}")
                
            elif user_input.lower() == 'docs':
                stats = query_system.get_system_stats()
                index_stats = stats.get('index_creation_stats', {})
                if index_stats and 'successful_documents' in index_stats:
                    docs = index_stats['successful_documents']
                    print(f"\n📚 Available Documents ({len(docs)} total):")
                    
                    # Group by topic and show doc_id
                    by_topic = {}
                    for doc in docs:
                        topic = doc.get('topic', 'Unknown')
                        if topic not in by_topic:
                            by_topic[topic] = []
                        by_topic[topic].append(doc)
                    
                    for topic, topic_docs in by_topic.items():
                        print(f"\n🏷️ {topic}:")
                        for doc in topic_docs[:3]:  # Show first 3
                            print(f"  🆔 {doc.get('document_id', 'Unknown')} - {doc.get('filename', 'Unknown')}")
                        if len(topic_docs) > 3:
                            print(f"  ... and {len(topic_docs) - 3} more")
                            
            elif user_input.lower().startswith('details '):
                setting = user_input.split(' ', 1)[1].strip().lower()
                if setting in ['on', 'true']:
                    show_details = True
                    print("✅ Detailed output enabled")
                elif setting in ['off', 'false']:
                    show_details = False
                    print("✅ Detailed output disabled")
                    
            elif user_input.lower() == 'json':
                if last_result:
                    json_output = convert_query_result_to_json(last_result)
                    print("\n📄 JSON Output:")
                    print("-" * 60)
                    print(json_output)
                    print("-" * 60)
                else:
                    print("❌ No results to show - execute a query first")
                    
            elif user_input.lower().startswith('save '):
                if last_result:
                    filename = user_input.split(' ', 1)[1].strip()
                    if not filename.endswith('.json'):
                        filename += '.json'
                    
                    success = save_query_results_json(last_result, filename)
                    if not success:
                        print(f"❌ Failed to save results")
                else:
                    print("❌ No results to save - execute a query first")
                    
            elif user_input:
                # Execute search
                result = query_system.query(
                    question=user_input,
                    k=5,
                    include_metadata=False  # Don't include full metadata for simplified output
                )
                
                last_result = result  # Store for JSON export
                
                # Display simplified results
                display_search_results(result, show_details)
                
                # Show simple summary
                summary = create_simple_query_summary(result)
                print(f"\n📊 Summary:")
                print(f"   📄 Found {summary['unique_documents_count']} unique documents")
                print(f"   🏷️ Topics: {', '.join(summary['topics_found'])}")
                
                # Show document list with doc_id
                if summary['documents_found']:
                    print(f"   📚 Documents:")
                    for doc in summary['documents_found']:
                        print(f"      🆔 {doc['doc_id']}: {doc['title']} ({doc['topic']})")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Search session ended!")

async def main():
    """Updated main function with simplified output"""
    parser = argparse.ArgumentParser(description='Vietnamese Document Query System')
    parser.add_argument('--index-path', default='storage/index',
                       help='Path to load vector index from')
    parser.add_argument('--query', help='Single query to execute')
    parser.add_argument('--k', type=int, default=5, help='Number of results')
    parser.add_argument('--simple', action='store_true', 
                       help='Use simplified output (default)')
    parser.add_argument('--detailed', action='store_true',
                       help='Use detailed output')
    parser.add_argument('--interactive', action='store_true',
                       help='Start interactive search session')
    parser.add_argument('--config-preset', default='development',
                    choices=['development', 'production', 'high_memory', 'low_memory'],
                    help='Configuration preset')
    parser.add_argument('--disable-hybrid', action='store_true',
                    help='Disable hybrid search')
    args = parser.parse_args()
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

    
    # Execute single query
    if args.query:
        result = query_system.query(
            question=args.query,
            k=args.k,
            include_metadata=args.detailed
        )
        
        if args.simple or not args.detailed:
            display_search_results(result, args.detailed)
        else:
            display_search_results(result, args.detailed)
        return
    
    # Start interactive session
    if args.interactive or not args.query:
        interactive_search_session_simplified(query_system)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())