# retrieval/retrieval_qdrant.py - Fixed version with proper UUID point IDs

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import pickle
import os
from abc import ABC, abstractmethod
from processing.chunker import Chunk
from processing.embedder import VietnameseEmbedder
from utils.utils import estimate_tokens
from retrieval.vector_db_qdrant import VectorStore

import logging
from datetime import datetime
import uuid
import hashlib

# Qdrant imports
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, Match, MatchValue, CollectionInfo,
    HnswConfigDiff, OptimizersConfigDiff
)

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    chunk_id: str
    score: float
    chunk: Chunk
    parent_chunk: Optional[Chunk] = None
    related_chunks: List[Chunk] = None
    context_metadata: Dict = None
    
    def __post_init__(self):
        if self.related_chunks is None:
            self.related_chunks = []
        if self.context_metadata is None:
            self.context_metadata = {}

# Keep the same HierarchicalRetriever class - no changes needed
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
        hierarchy_path = metadata.get('hierarchy_path', '')
        if hierarchy_path:
            header_parts.append(f"**📍 Vị trí:** {hierarchy_path}")
        
        # Administrative structure
        admin_info = metadata.get('administrative_info', {})
        if admin_info:
            admin_structure = self._format_admin_structure(admin_info)
            if admin_structure:
                header_parts.append(f"**🏛️ Cấu trúc hành chính:** {admin_structure}")
        
        # Content type and structure info
        structure_info = f"Loại: {chunk.chunk_type}, Cấp: {metadata.get('chunk_level', 'unknown')}"
        header_parts.append(f"**📑 Thông tin cấu trúc:** {structure_info}")
        
        # Table information if applicable
        if 'table' in chunk.chunk_type:
            table_info = metadata.get('table_info', {})
            if table_info.get('row_count') and table_info.get('column_count'):
                table_details = f"{table_info['row_count']} hàng × {table_info['column_count']} cột"
                header_parts.append(f"**📊 Thông tin bảng:** {table_details}")
        
        return "\n".join(header_parts)
    
    def _format_admin_structure(self, admin_info: Dict) -> str:
        """Format administrative structure information"""
        parts = []
        if admin_info.get('section'):
            parts.append(f"Phần {admin_info['section']}")
        if admin_info.get('chapter'):
            parts.append(f"Chương {admin_info['chapter']}")
        if admin_info.get('article'):
            parts.append(f"Điều {admin_info['article']}")
        if admin_info.get('point'):
            parts.append(f"Điểm {admin_info['point']}")
        return " > ".join(parts) if parts else ""
    
    def _create_parent_context_section(self, parent_chunk: Chunk) -> str:
        """Create parent context section"""
        parent_parts = []
        
        parent_parts.append("**👆 BỐI CẢNH TỪ PHẦN CHÍNH:**")
        
        # Parent metadata
        parent_metadata = parent_chunk.metadata
        parent_title = parent_metadata.get('section_title', 'Không có tiêu đề')
        
        if parent_title:
            parent_parts.append(f"**📌 Tiêu đề phần chính:** {parent_title}")
        
        # Parent content preview
        parent_content = parent_chunk.content
        if len(parent_content) > 400:
            parent_content_preview = parent_content[:400] + "\n[... nội dung phần chính còn tiếp tục ...]"
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
            
            # Short preview of related content
            related_preview = related_chunk.content[:200] + "..." if len(related_chunk.content) > 200 else related_chunk.content
            
            related_section = f"  **{i+1}. {related_title}**\n    {related_preview}"
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
        
        return "\n".join(metadata_parts) if metadata_parts else ""
    
    def _truncate_context_block(self, context_block: str, max_tokens: int) -> str:
        """Truncate context block to fit within token limit"""
        words = context_block.split()
        estimated_words = int(max_tokens * 0.8)  # Conservative estimate
        
        if len(words) > estimated_words:
            truncated_content = ' '.join(words[:estimated_words])
            return truncated_content + "\n\n[... nội dung bị cắt ngắn do giới hạn độ dài ...]"
        
        return context_block