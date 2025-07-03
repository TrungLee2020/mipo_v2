# retriever.py with improved parent-child context retrieval

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import faiss
import pickle
import os
from abc import ABC, abstractmethod
from processing.chunker import Chunk
from processing.embedder import VietnameseEmbedder
from utils.utils import estimate_tokens
import logging

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    chunk: Chunk
    parent_chunk: Optional[Chunk] = None
    related_chunks: List[Chunk] = None  # Add related chunks (siblings, etc.)
    context_metadata: Dict = None  # Enhanced context information
    
    def __post_init__(self):
        if self.related_chunks is None:
            self.related_chunks = []
        if self.context_metadata is None:
            self.context_metadata = {}

class VectorStore(ABC):
    """Abstract vector store interface"""
    
    @abstractmethod
    def add_embeddings(self, embeddings: Dict[str, np.ndarray], chunks: List[Chunk]):
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int, filter_dict: Dict = None) -> List[RetrievalResult]:
        pass
    
    @abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        pass
    
    @abstractmethod
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, Chunk]:
        pass
    
    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        pass
    
    @abstractmethod
    def save_index(self, path: str) -> bool:
        pass
    
    @abstractmethod
    def load_index(self, path: str) -> bool:
        pass

class FAISSVectorStore(VectorStore):
    """Enhanced FAISS-based vector store with improved parent-child handling"""
    
    def __init__(self, 
                 dimension: int = 768,
                 index_type: str = 'hnsw',
                 m: int = 32,
                 ef_construction: int = 200,
                 ef_search: int = 128,
                 metric: str = 'cosine'):
        """Initialize FAISS vector store"""
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Create index based on type
        self.index = self._create_index()
        
        # Mappings
        self.chunk_map = {}  # chunk_id -> Chunk
        self.id_map = {}     # faiss_id -> chunk_id
        self.reverse_id_map = {}  # chunk_id -> faiss_id
        self.embedding_map = {}  # chunk_id -> embedding
        self.next_id = 0
        self.deleted_ids = set()
        
        # Enhanced parent-child mappings
        self.parent_to_children = {}  # parent_id -> [child_ids]
        self.child_to_parent = {}     # child_id -> parent_id
        
        logger.info(f"Initialized {index_type.upper()} index with dimension {dimension}")
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration"""
        if self.index_type == 'hnsw':
            if self.metric == 'cosine':
                index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
            elif self.metric == 'l2':
                index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_L2)
            else:  # ip
                index = faiss.IndexHNSWFlat(self.dimension, self.m, faiss.METRIC_INNER_PRODUCT)
            
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            
        elif self.index_type == 'ivf':
            nlist = 100
            quantizer = faiss.IndexFlatIP(self.dimension) if self.metric != 'l2' else faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
        else:  # flat
            if self.metric == 'cosine' or self.metric == 'ip':
                index = faiss.IndexFlatIP(self.dimension)
            else:  # l2
                index = faiss.IndexFlatL2(self.dimension)
        
        return index
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity"""
        if self.metric == 'cosine':
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm
        return embedding
    
    def add_embeddings(self, embeddings: Dict[str, np.ndarray], chunks: List[Chunk]):
        """Add embeddings to FAISS index with enhanced parent-child mapping"""
        chunk_dict = {chunk.id: chunk for chunk in chunks}
        
        embeddings_list = []
        chunk_ids = []
        faiss_ids = []
        
        for chunk_id, embedding in embeddings.items():
            if chunk_id in chunk_dict and chunk_id not in self.chunk_map:
                chunk = chunk_dict[chunk_id]
                
                # Normalize embedding if needed
                normalized_embedding = self._normalize_embedding(embedding)
                embeddings_list.append(normalized_embedding)
                chunk_ids.append(chunk_id)
                
                # Reuse deleted ID or create new one
                if self.deleted_ids:
                    faiss_id = self.deleted_ids.pop()
                else:
                    faiss_id = self.next_id
                    self.next_id += 1
                    
                faiss_ids.append(faiss_id)
                
                # Update mappings
                self.chunk_map[chunk_id] = chunk
                self.id_map[faiss_id] = chunk_id
                self.reverse_id_map[chunk_id] = faiss_id
                self.embedding_map[chunk_id] = normalized_embedding
                
                # Update parent-child relationships
                self._update_parent_child_mappings(chunk)
        
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype('float32')
            
            # Train index if needed (for IVF)
            if self.index_type == 'ivf' and not self.index.is_trained:
                if len(embeddings_list) >= 100:
                    self.index.train(embeddings_array)
                    logger.info("Trained IVF index")
                else:
                    logger.warning("Not enough data to train IVF index, need at least 100 samples")
                    return
            
            # Add to index
            if self.index_type == 'ivf' and self.index.is_trained:
                self.index.add(embeddings_array)
            elif self.index_type != 'ivf':
                self.index.add(embeddings_array)
            
            logger.info(f"Added {len(embeddings_list)} embeddings to {self.index_type.upper()} index")
    
    def _update_parent_child_mappings(self, chunk: Chunk):
        """Update parent-child relationship mappings"""
        # Update parent -> children mapping
        if chunk.chunk_type in ['parent', 'table_parent'] and chunk.child_ids:
            self.parent_to_children[chunk.id] = chunk.child_ids
            
            # Update child -> parent mapping
            for child_id in chunk.child_ids:
                self.child_to_parent[child_id] = chunk.id
        
        # Update child -> parent mapping
        if chunk.parent_id:
            self.child_to_parent[chunk.id] = chunk.parent_id
            
            # Update parent -> children mapping
            if chunk.parent_id not in self.parent_to_children:
                self.parent_to_children[chunk.parent_id] = []
            if chunk.id not in self.parent_to_children[chunk.parent_id]:
                self.parent_to_children[chunk.parent_id].append(chunk.id)
    
    def search(self, query_embedding: np.ndarray, k: int, filter_dict: Dict = None) -> List[RetrievalResult]:
        """Search for similar chunks with enhanced parent-child context"""
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Normalize query embedding
        query_embedding = self._normalize_embedding(query_embedding)
        
        # Set search parameters for HNSW
        if self.index_type == 'hnsw':
            self.index.hnsw.efSearch = max(self.ef_search, k * 2)
        
        # Search in FAISS - get more candidates for context enrichment
        search_k = min(k * 5, self.index.ntotal)  # Get more candidates
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            search_k
        )
        
        results = []
        processed_chunks = set()  # Track processed chunks to avoid duplicates
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            chunk_id = self.id_map.get(idx)
            if not chunk_id or chunk_id not in self.chunk_map:
                continue
            
            chunk = self.chunk_map[chunk_id]
            
            # Apply filters
            if filter_dict and not self._matches_filter(chunk, filter_dict):
                continue
            
            # Skip if already processed (might happen with parent-child pairs)
            if chunk_id in processed_chunks:
                continue
            
            # Convert distance to similarity score
            if self.metric in ['cosine', 'ip']:
                similarity_score = float(score)
            else:  # L2 distance - convert to similarity
                similarity_score = 1.0 / (1.0 + float(score))
            
            # Create enhanced result with parent-child context
            enhanced_result = self._create_enhanced_result(chunk, similarity_score)
            
            if enhanced_result:
                results.append(enhanced_result)
                processed_chunks.add(chunk_id)
                
                # Mark related chunks as processed to avoid duplicates
                if enhanced_result.parent_chunk:
                    processed_chunks.add(enhanced_result.parent_chunk.id)
                for related_chunk in enhanced_result.related_chunks:
                    processed_chunks.add(related_chunk.id)
                
                if len(results) >= k:
                    break
        
        return results
    
    def _create_enhanced_result(self, chunk: Chunk, score: float) -> Optional[RetrievalResult]:
        """Create enhanced retrieval result with full parent-child context"""
        parent_chunk = None
        related_chunks = []
        context_metadata = {}
        
        # Get parent chunk if this is a child
        if chunk.parent_id and chunk.parent_id in self.chunk_map:
            parent_chunk = self.chunk_map[chunk.parent_id]
            
            # Add parent's metadata to context
            context_metadata['has_parent'] = True
            context_metadata['parent_metadata'] = parent_chunk.metadata
        
        # Get related chunks (siblings if this is a child, children if this is a parent)
        if chunk.chunk_type in ['child', 'table_child'] and chunk.parent_id:
            # Get sibling chunks
            sibling_ids = self.parent_to_children.get(chunk.parent_id, [])
            for sibling_id in sibling_ids:
                if sibling_id != chunk.id and sibling_id in self.chunk_map:
                    related_chunks.append(self.chunk_map[sibling_id])
            
            context_metadata['sibling_count'] = len(related_chunks)
            
        elif chunk.chunk_type in ['parent', 'table_parent']:
            # Get child chunks
            child_ids = self.parent_to_children.get(chunk.id, [])
            for child_id in child_ids:
                if child_id in self.chunk_map:
                    related_chunks.append(self.chunk_map[child_id])
            
            context_metadata['child_count'] = len(related_chunks)
        
        # Add hierarchy context metadata
        context_metadata.update(self._extract_hierarchy_context(chunk, parent_chunk))
        
        return RetrievalResult(
            chunk_id=chunk.id,
            score=score,
            chunk=chunk,
            parent_chunk=parent_chunk,
            related_chunks=related_chunks,
            context_metadata=context_metadata
        )
    
    def _extract_hierarchy_context(self, chunk: Chunk, parent_chunk: Optional[Chunk]) -> Dict:
        """Extract hierarchical context information"""
        context = {
            'chunk_type': chunk.chunk_type,
            'chunk_level': chunk.metadata.get('chunk_level', 'unknown'),
            'section_title': chunk.metadata.get('section_title', ''),
            'hierarchy_path': chunk.metadata.get('hierarchy_path', ''),
            'document_title': chunk.metadata.get('document_title', ''),
            'administrative_info': chunk.metadata.get('administrative_info', {}),
            'header_metadata': chunk.metadata.get('header_metadata', {}),
            'breadcrumb': chunk.metadata.get('breadcrumb', [])
        }
        
        # Add parent context if available
        if parent_chunk:
            context['parent_hierarchy_path'] = parent_chunk.metadata.get('hierarchy_path', '')
            context['parent_section_title'] = parent_chunk.metadata.get('section_title', '')
            context['parent_administrative_info'] = parent_chunk.metadata.get('administrative_info', {})
        
        return context
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        return self.chunk_map.get(chunk_id)
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, Chunk]:
        """Get multiple chunks by IDs"""
        result = {}
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_map:
                result[chunk_id] = self.chunk_map[chunk_id]
        return result
    
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks from the index"""
        try:
            deleted_count = 0
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_map:
                    # Clean up parent-child mappings
                    self._cleanup_parent_child_mappings(chunk_id)
                    
                    # Get FAISS ID
                    faiss_id = self.reverse_id_map.get(chunk_id)
                    if faiss_id is not None:
                        # Remove from mappings
                        del self.chunk_map[chunk_id]
                        del self.id_map[faiss_id]
                        del self.reverse_id_map[chunk_id]
                        if chunk_id in self.embedding_map:
                            del self.embedding_map[chunk_id]
                        
                        # Mark ID as deleted for reuse
                        self.deleted_ids.add(faiss_id)
                        deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} chunks from index")
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False
    
    def _cleanup_parent_child_mappings(self, chunk_id: str):
        """Clean up parent-child mappings when deleting a chunk"""
        # Remove from child_to_parent mapping
        if chunk_id in self.child_to_parent:
            parent_id = self.child_to_parent[chunk_id]
            del self.child_to_parent[chunk_id]
            
            # Remove from parent's children list
            if parent_id in self.parent_to_children:
                self.parent_to_children[parent_id] = [
                    cid for cid in self.parent_to_children[parent_id] 
                    if cid != chunk_id
                ]
                if not self.parent_to_children[parent_id]:
                    del self.parent_to_children[parent_id]
        
        # Remove from parent_to_children mapping
        if chunk_id in self.parent_to_children:
            child_ids = self.parent_to_children[chunk_id]
            del self.parent_to_children[chunk_id]
            
            # Remove children from child_to_parent mapping
            for child_id in child_ids:
                if child_id in self.child_to_parent:
                    del self.child_to_parent[child_id]
    
    def rebuild_index(self):
        """Rebuild index to reclaim space from deleted vectors"""
        if not self.chunk_map:
            logger.warning("No chunks to rebuild index")
            return
        
        logger.info("Rebuilding index to reclaim space...")
        
        # Save current data
        chunks = list(self.chunk_map.values())
        embeddings = {chunk_id: self.embedding_map[chunk_id] 
                     for chunk_id in self.chunk_map.keys()}
        
        # Reset index and mappings
        self.index = self._create_index()
        self.chunk_map.clear()
        self.id_map.clear()
        self.reverse_id_map.clear()
        self.embedding_map.clear()
        self.parent_to_children.clear()
        self.child_to_parent.clear()
        self.next_id = 0
        self.deleted_ids.clear()
        
        # Re-add all chunks
        self.add_embeddings(embeddings, chunks)
        logger.info("Index rebuilt successfully")
    
    def save_index(self, path: str) -> bool:
        """Save index and metadata to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.index")
            
            # Save metadata
            metadata = {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'metric': self.metric,
                'm': self.m,
                'ef_construction': self.ef_construction,
                'ef_search': self.ef_search,
                'chunk_map': self.chunk_map,
                'id_map': self.id_map,
                'reverse_id_map': self.reverse_id_map,
                'embedding_map': self.embedding_map,
                'parent_to_children': self.parent_to_children,
                'child_to_parent': self.child_to_parent,
                'next_id': self.next_id,
                'deleted_ids': self.deleted_ids
            }
            
            with open(f"{path}.metadata", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved index to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def load_index(self, path: str) -> bool:
        """Load index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.index")
            
            # Load metadata
            with open(f"{path}.metadata", 'rb') as f:
                metadata = pickle.load(f)
            
            # Restore configuration
            self.dimension = metadata['dimension']
            self.index_type = metadata['index_type']
            self.metric = metadata['metric']
            self.m = metadata['m']
            self.ef_construction = metadata['ef_construction']
            self.ef_search = metadata['ef_search']
            
            # Restore mappings
            self.chunk_map = metadata['chunk_map']
            self.id_map = metadata['id_map']
            self.reverse_id_map = metadata['reverse_id_map']
            self.embedding_map = metadata['embedding_map']
            self.parent_to_children = metadata.get('parent_to_children', {})
            self.child_to_parent = metadata.get('child_to_parent', {})
            self.next_id = metadata['next_id']
            self.deleted_ids = metadata['deleted_ids']
            
            # Set HNSW search parameters if needed
            if self.index_type == 'hnsw':
                self.index.hnsw.efSearch = self.ef_search
            
            logger.info(f"Loaded index from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'active_chunks': len(self.chunk_map),
            'deleted_chunks': len(self.deleted_ids),
            'next_id': self.next_id,
            'parent_child_relationships': len(self.parent_to_children),
            'child_chunks': len(self.child_to_parent)
        }
    
    def _matches_filter(self, chunk: Chunk, filter_dict: Dict) -> bool:
        """Check if chunk matches filter criteria"""
        for key, value in filter_dict.items():
            if key == 'chunk_type':
                if isinstance(value, list):
                    if chunk.chunk_type not in value:
                        return False
                elif chunk.chunk_type != value:
                    return False
            elif key in chunk.metadata:
                if chunk.metadata[key] != value:
                    return False
        return True

class HierarchicalRetriever:
    """Enhanced hierarchical retrieval system with improved parent-child context"""
    
    def __init__(self, vector_store: VectorStore, embedder: VietnameseEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder
    
    def retrieve(self, query: str, k: int = 5, retrieval_strategy: str = 'enhanced_hierarchical') -> List[RetrievalResult]:
        """Main retrieval function with enhanced strategies"""
        query_embedding = self.embedder.embed_query(query)
        
        if retrieval_strategy == 'enhanced_hierarchical':
            return self._enhanced_hierarchical_retrieve(query_embedding, k)
        elif retrieval_strategy == 'hierarchical':
            return self._hierarchical_retrieve(query_embedding, k)
        elif retrieval_strategy == 'table_aware':
            return self._table_aware_retrieve(query, query_embedding, k)
        else:
            return self._simple_retrieve(query_embedding, k)
    
    def _enhanced_hierarchical_retrieve(self, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
        """Enhanced hierarchical retrieval with full parent-child context"""
        # Get initial results from vector store (already enhanced with context)
        initial_results = self.vector_store.search(query_embedding, k * 2)
        
        # Further enhance results with additional context processing
        enhanced_results = []
        seen_chunks = set()
        
        for result in initial_results:
            if result.chunk_id in seen_chunks:
                continue
            
            # Mark all related chunks as seen to avoid duplicates
            seen_chunks.add(result.chunk_id)
            if result.parent_chunk:
                seen_chunks.add(result.parent_chunk.id)
            for related_chunk in result.related_chunks:
                seen_chunks.add(related_chunk.id)
            
            # Add comprehensive context metadata
            result.context_metadata.update({
                'retrieval_strategy': 'enhanced_hierarchical',
                'query_relevance_context': self._analyze_query_relevance(result),
                'structural_context': self._extract_structural_context(result),
                'content_summary': self._create_content_summary(result)
            })
            
            enhanced_results.append(result)
            
            if len(enhanced_results) >= k:
                break
        
        return enhanced_results
    
    def _analyze_query_relevance(self, result: RetrievalResult) -> Dict:
        """Analyze why this result is relevant to the query"""
        relevance_context = {
            'primary_match': result.chunk.chunk_type,
            'score': result.score,
            'has_parent_context': result.parent_chunk is not None,
            'has_related_content': len(result.related_chunks) > 0
        }
        
        # Analyze content type relevance
        if 'table' in result.chunk.chunk_type:
            relevance_context['content_type'] = 'tabular_data'
            table_info = result.chunk.metadata.get('table_info', {})
            if table_info:
                relevance_context['table_details'] = {
                    'rows': table_info.get('row_count', 0),
                    'columns': table_info.get('column_count', 0),
                    'headers': table_info.get('headers', [])
                }
        else:
            relevance_context['content_type'] = 'textual_content'
        
        return relevance_context
    
    def _extract_structural_context(self, result: RetrievalResult) -> Dict:
        """Extract structural context information"""
        structural_context = {
            'hierarchy_level': result.chunk.metadata.get('chunk_level', 'unknown'),
            'document_structure': {
                'section': result.chunk.metadata.get('section_title', ''),
                'hierarchy_path': result.chunk.metadata.get('hierarchy_path', ''),
                'admin_structure': result.chunk.metadata.get('administrative_info', {})
            }
        }
        
        # Add parent structural context
        if result.parent_chunk:
            structural_context['parent_structure'] = {
                'section': result.parent_chunk.metadata.get('section_title', ''),
                'hierarchy_path': result.parent_chunk.metadata.get('hierarchy_path', ''),
                'content_overview': result.parent_chunk.content[:200] + "..."
            }
        
        # Add breadcrumb for navigation
        breadcrumb = result.chunk.metadata.get('breadcrumb', [])
        if breadcrumb:
            structural_context['navigation_path'] = [
                item.get('title', '') for item in breadcrumb
            ]
        
        return structural_context
    
    def _create_content_summary(self, result: RetrievalResult) -> Dict:
        """Create a summary of the content and its context"""
        content_summary = {
            'primary_content_length': len(result.chunk.content),
            'primary_content_preview': result.chunk.content[:300] + "..." if len(result.chunk.content) > 300 else result.chunk.content
        }
        
        # Add parent content summary
        if result.parent_chunk:
            content_summary['parent_content_overview'] = {
                'length': len(result.parent_chunk.content),
                'preview': result.parent_chunk.content[:200] + "..." if len(result.parent_chunk.content) > 200 else result.parent_chunk.content
            }
        
        # Add related content overview
        if result.related_chunks:
            content_summary['related_content_count'] = len(result.related_chunks)
            content_summary['related_content_types'] = list(set(
                chunk.chunk_type for chunk in result.related_chunks
            ))
        
        return content_summary
    
    def _hierarchical_retrieve(self, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
        """Original hierarchical retrieval method"""
        return self.vector_store.search(query_embedding, k)
    
    def _table_aware_retrieve(self, query: str, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
        """Table-aware retrieval"""
        query_type = self._classify_query_type(query)
        
        if query_type == 'table_query':
            # Prioritize table chunks
            results = self.vector_store.search(
                query_embedding,
                k,
                filter_dict={'chunk_type': ['table_child', 'table_parent']}
            )
            
            if not results:
                # Fallback to general search
                results = self.vector_store.search(query_embedding, k)
            
            return results
        else:
            return self._enhanced_hierarchical_retrieve(query_embedding, k)
    
    def _simple_retrieve(self, query_embedding: np.ndarray, k: int) -> List[RetrievalResult]:
        """Simple similarity search"""
        return self.vector_store.search(query_embedding, k)
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for appropriate retrieval strategy"""
        query_lower = query.lower()
        
        table_keywords = [
            'bảng', 'biểu', 'danh sách', 'thống kê', 'số liệu',
            'cột', 'hàng', 'dữ liệu', 'tỷ lệ', 'phần trăm'
        ]
        
        if any(keyword in query_lower for keyword in table_keywords):
            return 'table_query'
        
        return 'text_query'
    
    def format_context(self, results: List[RetrievalResult], max_tokens: int = 4096) -> str:
        """Enhanced context formatting with full parent-child context"""
        context_parts = []
        current_tokens = 0
        
        for i, result in enumerate(results):
            if current_tokens >= max_tokens:
                break
            
            # Create comprehensive context block
            context_block = self._create_enhanced_context_block(result, i + 1)
            block_tokens = estimate_tokens(context_block)
            
            if current_tokens + block_tokens > max_tokens:
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 300:  # Only add if meaningful space left
                    truncated_block = self._truncate_context_block(context_block, remaining_tokens)
                    context_parts.append(truncated_block)
                break
            
            context_parts.append(context_block)
            current_tokens += block_tokens
        
        return "\n\n" + "="*80 + "\n\n".join(context_parts)
    
    def _create_enhanced_context_block(self, result: RetrievalResult, source_number: int) -> str:
        """Create enhanced context block with full parent-child information"""
        block_parts = []
        
        # Main header with comprehensive metadata
        header = self._create_comprehensive_header(result, source_number)
        block_parts.append(header)
        
        # Primary content
        primary_content = f"**📄 Nội dung chính:**\n{result.chunk.content}"
        block_parts.append(primary_content)
        
        # Parent context if available
        if result.parent_chunk:
            parent_context = self._create_parent_context_section(result.parent_chunk)
            block_parts.append(parent_context)
        
        # Related content context
        if result.related_chunks:
            related_context = self._create_related_context_section(result.related_chunks)
            block_parts.append(related_context)
        
        # Additional metadata context
        metadata_context = self._create_metadata_context_section(result)
        if metadata_context:
            block_parts.append(metadata_context)
        
        return "\n\n".join(block_parts)
    
    def _create_comprehensive_header(self, result: RetrievalResult, source_number: int) -> str:
        """Create comprehensive header with all available metadata"""
        header_parts = []
        
        # Basic source info
        header_parts.append(f"**📄 Nguồn {source_number}** (Độ liên quan: {result.score:.3f})")
        
        # Document and hierarchy context
        chunk = result.chunk
        metadata = chunk.metadata
        
        # Document info
        doc_title = metadata.get('document_title', 'Không rõ tài liệu')
        header_parts.append(f"**📋 Tài liệu:** {doc_title}")
        
        # Enhanced hierarchy path
        hierarchy_path = self._extract_full_hierarchy_path(result)
        if hierarchy_path:
            header_parts.append(f"**📍 Vị trí đầy đủ:** {hierarchy_path}")
        
        # Administrative structure
        admin_structure = self._extract_administrative_structure(result)
        if admin_structure:
            header_parts.append(f"**🏛️ Cấu trúc hành chính:** {admin_structure}")
        
        # Breadcrumb navigation
        breadcrumb = self._extract_breadcrumb_navigation(result)
        if breadcrumb:
            header_parts.append(f"**🧭 Đường dẫn điều hướng:** {breadcrumb}")
        
        # Content type and structure info
        structure_info = self._extract_structure_information(result)
        header_parts.append(f"**📑 Thông tin cấu trúc:** {structure_info}")
        
        # Table information if applicable
        table_info = self._extract_table_information(result)
        if table_info:
            header_parts.append(f"**📊 Thông tin bảng:** {table_info}")
        
        # Parent-child relationship info
        relationship_info = self._extract_relationship_information(result)
        if relationship_info:
            header_parts.append(f"**🔗 Mối quan hệ:** {relationship_info}")
        
        return "\n".join(header_parts)
    
    def _create_parent_context_section(self, parent_chunk: Chunk) -> str:
        """Create parent context section"""
        parent_parts = []
        
        parent_parts.append("**👆 BỐI CẢNH TỪ PHẦN CHÍNH:**")
        
        # Parent metadata
        parent_metadata = parent_chunk.metadata
        parent_title = parent_metadata.get('section_title', 'Không có tiêu đề')
        parent_hierarchy = parent_metadata.get('hierarchy_path', '')
        
        if parent_title:
            parent_parts.append(f"**📌 Tiêu đề phần chính:** {parent_title}")
        
        if parent_hierarchy:
            parent_parts.append(f"**📍 Vị trí phần chính:** {parent_hierarchy}")
        
        # Parent administrative info
        parent_admin = parent_metadata.get('administrative_info', {})
        if parent_admin:
            admin_details = []
            if parent_admin.get('section_full'):
                admin_details.append(parent_admin['section_full'])
            elif parent_admin.get('section'):
                admin_details.append(f"Phần {parent_admin['section']}")
            
            if parent_admin.get('chapter_full'):
                admin_details.append(parent_admin['chapter_full'])
            elif parent_admin.get('chapter'):
                admin_details.append(f"Chương {parent_admin['chapter']}")
            
            if parent_admin.get('article_full'):
                admin_details.append(parent_admin['article_full'])
            elif parent_admin.get('article'):
                admin_details.append(f"Điều {parent_admin['article']}")
            
            if admin_details:
                parent_parts.append(f"**🏛️ Cấu trúc hành chính:** {' > '.join(admin_details)}")
        
        # Parent content preview
        parent_content = parent_chunk.content
        if len(parent_content) > 500:
            parent_content_preview = parent_content[:500] + "\n[... nội dung phần chính còn tiếp tục ...]"
        else:
            parent_content_preview = parent_content
        
        parent_parts.append(f"**📄 Nội dung phần chính:**\n{parent_content_preview}")
        
        return "\n".join(parent_parts)
    
    def _create_related_context_section(self, related_chunks: List[Chunk]) -> str:
        """Create related content context section"""
        if not related_chunks:
            return ""
        
        related_parts = []
        related_parts.append(f"**🔗 NỘI DUNG LIÊN QUAN ({len(related_chunks)} phần):**")
        
        for i, related_chunk in enumerate(related_chunks[:3]):  # Limit to first 3 related chunks
            related_metadata = related_chunk.metadata
            related_title = related_metadata.get('section_title', f'Phần liên quan {i+1}')
            related_type = related_chunk.chunk_type
            
            # Short preview of related content
            related_preview = related_chunk.content[:200] + "..." if len(related_chunk.content) > 200 else related_chunk.content
            
            related_section = f"  **{i+1}. {related_title}** ({related_type})\n    {related_preview}"
            related_parts.append(related_section)
        
        if len(related_chunks) > 3:
            related_parts.append(f"    ... và {len(related_chunks) - 3} phần liên quan khác")
        
        return "\n".join(related_parts)
    
    def _create_metadata_context_section(self, result: RetrievalResult) -> str:
        """Create additional metadata context section"""
        context_metadata = result.context_metadata
        if not context_metadata:
            return ""
        
        metadata_parts = []
        
        # Query relevance context
        query_relevance = context_metadata.get('query_relevance_context', {})
        if query_relevance:
            content_type = query_relevance.get('content_type', 'unknown')
            type_display = {
                'tabular_data': 'Dữ liệu dạng bảng',
                'textual_content': 'Nội dung văn bản',
                'unknown': 'Không xác định'
            }.get(content_type, content_type)
            
            metadata_parts.append(f"**📊 Loại nội dung:** {type_display}")
            
            # Table details if available
            table_details = query_relevance.get('table_details', {})
            if table_details:
                rows = table_details.get('rows', 0)
                columns = table_details.get('columns', 0)
                headers = table_details.get('headers', [])
                metadata_parts.append(f"**📊 Chi tiết bảng:** {rows} hàng, {columns} cột")
                if headers:
                    headers_preview = ', '.join(headers[:5])
                    if len(headers) > 5:
                        headers_preview += f" (và {len(headers) - 5} cột khác)"
                    metadata_parts.append(f"**📊 Các cột:** {headers_preview}")
        
        # Structural context
        structural_context = context_metadata.get('structural_context', {})
        if structural_context:
            navigation_path = structural_context.get('navigation_path', [])
            if navigation_path:
                nav_display = " ➤ ".join(navigation_path)
                metadata_parts.append(f"**🧭 Đường dẫn điều hướng:** {nav_display}")
        
        return "\n".join(metadata_parts) if metadata_parts else ""
    
    def _extract_full_hierarchy_path(self, result: RetrievalResult) -> str:
        """Extract full hierarchy path including parent context"""
        paths = []
        
        # Get primary chunk path
        primary_path = result.chunk.metadata.get('hierarchy_path', '')
        if primary_path:
            paths.append(f"Chính: {primary_path}")
        
        # Get parent path if different and available
        if result.parent_chunk:
            parent_path = result.parent_chunk.metadata.get('hierarchy_path', '')
            if parent_path and parent_path != primary_path:
                paths.append(f"Phần chính: {parent_path}")
        
        # Get enhanced path from header metadata
        header_metadata = result.chunk.metadata.get('header_metadata', {})
        if header_metadata.get('full_path'):
            enhanced_path = header_metadata['full_path']
            if enhanced_path not in [primary_path, result.parent_chunk.metadata.get('hierarchy_path', '') if result.parent_chunk else None]:
                paths.append(f"Chi tiết: {enhanced_path}")
        
        return " | ".join(paths) if paths else ""
    
    def _extract_administrative_structure(self, result: RetrievalResult) -> str:
        """Extract administrative structure information"""
        admin_parts = []
        
        # Check primary chunk
        admin_info = result.chunk.metadata.get('administrative_info', {})
        header_admin = result.chunk.metadata.get('header_metadata', {}).get('header_administrative_info', {})
        
        # Use enhanced admin info if available
        if header_admin:
            if header_admin.get('section_full'):
                admin_parts.append(header_admin['section_full'])
            if header_admin.get('chapter_full'):
                admin_parts.append(header_admin['chapter_full'])
            if header_admin.get('article_full'):
                admin_parts.append(header_admin['article_full'])
        elif admin_info:
            if admin_info.get('section'):
                admin_parts.append(f"Phần {admin_info['section']}")
            if admin_info.get('chapter'):
                admin_parts.append(f"Chương {admin_info['chapter']}")
            if admin_info.get('article'):
                admin_parts.append(f"Điều {admin_info['article']}")
            if admin_info.get('point'):
                admin_parts.append(f"Điểm {admin_info['point']}")
        
        # Add parent administrative context if different
        if result.parent_chunk:
            parent_admin = result.parent_chunk.metadata.get('administrative_info', {})
            if parent_admin and parent_admin != admin_info:
                parent_parts = []
                if parent_admin.get('section'):
                    parent_parts.append(f"Phần {parent_admin['section']}")
                if parent_admin.get('chapter'):
                    parent_parts.append(f"Chương {parent_admin['chapter']}")
                
                if parent_parts:
                    admin_parts.extend([f"[Thuộc: {' > '.join(parent_parts)}]"])
        
        return " > ".join(admin_parts) if admin_parts else ""
    
    def _extract_breadcrumb_navigation(self, result: RetrievalResult) -> str:
        """Extract breadcrumb navigation"""
        breadcrumb = result.chunk.metadata.get('breadcrumb', [])
        if breadcrumb:
            breadcrumb_titles = []
            for item in breadcrumb:
                title = item.get('title', '')
                if len(title) > 50:
                    title = title[:50] + "..."
                breadcrumb_titles.append(title)
            return " ➤ ".join(breadcrumb_titles)
        return ""
    
    def _extract_structure_information(self, result: RetrievalResult) -> str:
        """Extract structure information"""
        chunk = result.chunk
        metadata = chunk.metadata
        
        info_parts = []
        
        # Chunk level and type
        chunk_level = metadata.get('chunk_level', 'unknown')
        chunk_type = chunk.chunk_type
        
        type_mapping = {
            'parent': 'Phần chính',
            'child': 'Phần con',
            'table_parent': 'Phần chính có bảng',
            'table_child': 'Phần con của bảng'
        }
        
        type_display = type_mapping.get(chunk_type, chunk_type)
        info_parts.append(f"Loại: {type_display}")
        info_parts.append(f"Cấp: {chunk_level}")
        
        # Content type
        content_type = metadata.get('content_type', 'text')
        if content_type != 'text':
            content_mapping = {
                'table_summary': 'Tóm tắt bảng',
                'table_data': 'Dữ liệu bảng'
            }
            info_parts.append(f"Nội dung: {content_mapping.get(content_type, content_type)}")
        
        # Section level
        section_level = metadata.get('section_level', 0)
        if section_level > 0:
            info_parts.append(f"Cấp mục: {section_level}")
        
        # Line range
        start_line = metadata.get('start_line', 0)
        end_line = metadata.get('end_line', 0)
        if start_line > 0 or end_line > 0:
            info_parts.append(f"Dòng: {start_line}-{end_line}")
        
        return " | ".join(info_parts)
    
    def _extract_table_information(self, result: RetrievalResult) -> str:
        """Extract table-specific information"""
        table_info = result.chunk.metadata.get('table_info', {})
        if not table_info:
            return ""
        
        info_parts = []
        
        # Basic table stats
        row_count = table_info.get('row_count', 0)
        column_count = table_info.get('column_count', 0)
        if row_count > 0 and column_count > 0:
            info_parts.append(f"{row_count} hàng × {column_count} cột")
        
        # Headers
        headers = table_info.get('headers', [])
        if headers:
            headers_preview = ', '.join(headers[:4])
            if len(headers) > 4:
                headers_preview += f" (và {len(headers) - 4} cột khác)"
            info_parts.append(f"Cột: {headers_preview}")
        
        # Table context
        table_context = table_info.get('table_header_context', {})
        if table_context and table_context.get('title'):
            info_parts.append(f"Thuộc: {table_context['title']}")
        
        # Chunk range for large tables
        chunk_start_row = table_info.get('chunk_start_row')
        chunk_end_row = table_info.get('chunk_end_row')
        if chunk_start_row is not None and chunk_end_row is not None:
            info_parts.append(f"Hàng {chunk_start_row + 1}-{chunk_end_row + 1}")
        
        return " | ".join(info_parts)
    
    def _extract_relationship_information(self, result: RetrievalResult) -> str:
        """Extract parent-child relationship information"""
        relationship_parts = []
        
        # Parent relationship
        if result.parent_chunk:
            parent_title = result.parent_chunk.metadata.get('section_title', 'Phần chính')
            relationship_parts.append(f"Thuộc: {parent_title}")
        
        # Related chunks information
        if result.related_chunks:
            related_count = len(result.related_chunks)
            if result.chunk.chunk_type in ['child', 'table_child']:
                relationship_parts.append(f"Có {related_count} phần anh em")
            else:
                relationship_parts.append(f"Có {related_count} phần con")
        
        # Hierarchy context
        context_metadata = result.context_metadata or {}
        if context_metadata.get('sibling_count', 0) > 0:
            sibling_count = context_metadata['sibling_count']
            relationship_parts.append(f"{sibling_count} phần cùng cấp")
        elif context_metadata.get('child_count', 0) > 0:
            child_count = context_metadata['child_count']
            relationship_parts.append(f"{child_count} phần con")
        
        return " | ".join(relationship_parts) if relationship_parts else ""
    
    def _truncate_context_block(self, context_block: str, max_tokens: int) -> str:
        """Truncate context block to fit within token limit"""
        words = context_block.split()
        estimated_words = int(max_tokens * 0.8)  # Conservative estimate
        
        if len(words) > estimated_words:
            truncated_content = ' '.join(words[:estimated_words])
            return truncated_content + "\n\n[... nội dung bị cắt ngắn do giới hạn độ dài ...]"
        
        return context_block
    
    def format_context_for_display(self, results: List[RetrievalResult]) -> List[Dict]:
        """Format retrieval results for display with enhanced parent-child context"""
        formatted_results = []
        
        for i, result in enumerate(results):
            chunk = result.chunk
            metadata = chunk.metadata
            
            # Create comprehensive result info
            result_info = {
                'source_number': i + 1,
                'score': result.score,
                'chunk_id': result.chunk_id,
                'content': chunk.content,
                'content_preview': chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content,
                
                # Enhanced document info
                'document_title': metadata.get('document_title', 'Không rõ'),
                'hierarchy_path': self._extract_full_hierarchy_path(result),
                'administrative_structure': self._extract_administrative_structure(result),
                'breadcrumb': self._extract_breadcrumb_navigation(result),
                
                # Enhanced chunk info
                'chunk_type': chunk.chunk_type,
                'chunk_level': metadata.get('chunk_level', 'unknown'),
                'content_type': metadata.get('content_type', 'text'),
                'section_title': metadata.get('section_title', ''),
                'section_level': metadata.get('section_level', 0),
                
                # Enhanced metadata
                'header_metadata': metadata.get('header_metadata', {}),
                'administrative_info': metadata.get('administrative_info', {}),
                'table_info': metadata.get('table_info', {}),
                'breadcrumb_list': metadata.get('breadcrumb', []),
                
                # Line numbers for reference
                'start_line': metadata.get('start_line', 0),
                'end_line': metadata.get('end_line', 0),
                
                # Enhanced parent-child context
                'has_parent': result.parent_chunk is not None,
                'parent_info': {},
                'related_chunks_info': [],
                'context_metadata': result.context_metadata or {}
            }
            
            # Add comprehensive parent information
            if result.parent_chunk:
                parent_metadata = result.parent_chunk.metadata
                result_info['parent_info'] = {
                    'chunk_id': result.parent_chunk.id,
                    'title': parent_metadata.get('section_title', ''),
                    'hierarchy_path': parent_metadata.get('hierarchy_path', ''),
                    'content_preview': result.parent_chunk.content[:300] + "..." if len(result.parent_chunk.content) > 300 else result.parent_chunk.content,
                    'chunk_type': result.parent_chunk.chunk_type,
                    'administrative_info': parent_metadata.get('administrative_info', {}),
                    'header_metadata': parent_metadata.get('header_metadata', {})
                }
            
            # Add comprehensive related chunks information
            for related_chunk in result.related_chunks:
                related_metadata = related_chunk.metadata
                related_info = {
                    'chunk_id': related_chunk.id,
                    'title': related_metadata.get('section_title', ''),
                    'chunk_type': related_chunk.chunk_type,
                    'content_preview': related_chunk.content[:200] + "..." if len(related_chunk.content) > 200 else related_chunk.content,
                    'hierarchy_path': related_metadata.get('hierarchy_path', '')
                }
                result_info['related_chunks_info'].append(related_info)
            
            formatted_results.append(result_info)
        
        return formatted_results