import os
import pickle
import logging
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path

# LlamaIndex imports for BM25
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from processing.chunker import Chunk

logger = logging.getLogger(__name__)


class BM25Manager:
    """Manager for BM25 retrieval using LlamaIndex"""
    
    def __init__(self, persist_path: str = "./indices/bm25"):
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.retriever = None
        self.chunk_map = {}  # chunk_id -> Chunk mapping
        self.node_map = {}   # node_id -> chunk_id mapping
        
    def build_index(self, chunks: List[Chunk]) -> bool:
        """Build BM25 index from chunks"""
        try:
            logger.info(f"Building BM25 index for {len(chunks)} chunks...")
            
            # Convert chunks to LlamaIndex nodes
            nodes = []
            self.chunk_map = {}
            self.node_map = {}
            
            for chunk in chunks:
                # Skip empty chunks
                if not chunk.content or not chunk.content.strip():
                    logger.warning(f"Skipping empty chunk: {chunk.id}")
                    continue

                # Adaptive chunking based on chunk type
                optimal_size = self._get_optimal_chunk_size(chunk)
                splitter = SentenceSplitter(
                    chunk_size=optimal_size, 
                    chunk_overlap=optimal_size // 8
                    )

                # Create LlamaIndex Document
                doc = Document(
                    text=chunk.content,
                    metadata={
                        'chunk_id': chunk.id,
                        'chunk_type': chunk.chunk_type,
                        'document_title': chunk.metadata.get('document_title', ''),
                        'section_title': chunk.metadata.get('section_title', ''),
                        # 'hierarchy_path': chunk.metadata.get('hierarchy_path', ''),
                        # 'original_chunk_size': len(chunk.content),
                        'content_density': self._calculate_content_density(chunk.content)
                    }
                )
                
                # Split into nodes if content is too long
                # splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
                # chunk_nodes = splitter.get_nodes_from_documents([doc])
                chunk_nodes = self._smart_split_content(doc, splitter)

                
                for i, node in enumerate(chunk_nodes):
                    node_id = f"{chunk.id}_bm25_{i}"
                    node.node_id = node_id
                    
                    # Enhanced metadata for BM25
                    node.metadata.update({
                        'node_index': i,
                        'total_nodes': len(chunk_nodes),
                        'bm25_weight': self._calculate_bm25_weight(node.text, chunk)
                    })
                    
                    self.node_map[node_id] = chunk.id
                    self.chunk_map[chunk.id] = chunk
                    nodes.append(node)
            
            # Create BM25 retriever
            self.retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=10,  # High number for fusion
                verbose=True
            )
            
            # Save index
            self._save_index()
            
            logger.info(f"✅ BM25 index built with {len(nodes)} nodes from {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building BM25 index: {e}")
            return False
    
    def _calculate_bm25_weight(self, text: str, chunk: Chunk) -> float:
        """Calculate BM25 weight based on content characteristics"""
        # Base weight
        base_weight = 1.0
        
        # Adjust based on chunk type
        chunk_type = chunk.chunk_type
        type_weights = {
            'header': 1.3,      # Headers are more important
            'paragraph': 1.0,   # Standard weight
            'list': 0.9,        # Lists slightly less important
            'code': 1.1,        # Code blocks important
            'table': 1.2        # Tables important
        }
        base_weight *= type_weights.get(chunk_type, 1.0)
        
        # Adjust based on content length
        text_length = len(text)
        if text_length < 100:
            base_weight *= 0.8  # Very short content less important
        elif text_length > 500:
            base_weight *= 1.2  # Longer content more important
        
        # Adjust based on content density (already calculated)
        content_density = chunk.metadata.get('content_density', 0.5)
        base_weight *= (0.8 + content_density * 0.4)  # Scale between 0.8-1.2
        
        # Adjust based on hierarchy level (if available)
        hierarchy_path = chunk.metadata.get('hierarchy_path', '')
        if hierarchy_path:
            level = hierarchy_path.count('/')
            if level <= 2:  # Top-level sections
                base_weight *= 1.1
            elif level >= 4:  # Deep nested sections
                base_weight *= 0.9
        
        # Ensure weight is in reasonable range
        return max(0.5, min(2.0, base_weight))

    def _get_optimal_chunk_size(self, chunk: Chunk) -> int:
        """Get optimal chunk size based on content type and length"""
        content_length = len(chunk.content)
        chunk_type = chunk.chunk_type

        # Nếu content_length <= 0 thì trả về 1 (hoặc giá trị tối thiểu hợp lệ)
        if content_length <= 0:
            return 1
        # Adaptive sizing
        if chunk_type == 'header':
            return min(512, content_length)
        elif chunk_type == 'paragraph':
            return min(768, max(256, content_length // 2))
        elif chunk_type == 'list':
            return min(512, max(128, content_length))
        else:
            # Default adaptive size
            if content_length < 300:
                return content_length
            elif content_length < 800:
                return 512
            else:
                return 1024

    def _smart_split_content(self, doc: Document, splitter: SentenceSplitter) -> List:
        """Smart content splitting with markdown awareness"""
        content = doc.text
        
        # Don't split very short content
        if len(content) < 200:
            from llama_index.core.schema import TextNode
            node = TextNode(
                text=content,
                metadata=doc.metadata
            )
            return [node]
        
        # Use sentence splitter for longer content
        return splitter.get_nodes_from_documents([doc])

    def _calculate_content_density(self, content: str) -> float:
        """Calculate content density for BM25 weighting"""
        words = content.split()
        if not words:
            return 0.0
        
        # Simple density metric
        unique_words = len(set(words))
        total_words = len(words)
        avg_word_length = sum(len(word) for word in words) / total_words
        
        return (unique_words / total_words) * (avg_word_length / 10)
    
    def load_index(self) -> bool:
        """Load BM25 index from disk"""
        try:
            index_file = self.persist_path / "bm25_index.pkl"
            mappings_file = self.persist_path / "bm25_mappings.pkl"
            
            if not (index_file.exists() and mappings_file.exists()):
                logger.info("No BM25 index found")
                return False
            
            # Load BM25 retriever
            self.retriever = BM25Retriever.from_persist_dir(str(self.persist_path))
            
            # Load mappings
            with open(mappings_file, 'rb') as f:
                mappings = pickle.load(f)
                self.chunk_map = mappings['chunk_map']
                self.node_map = mappings['node_map']
            
            logger.info(f"✅ Loaded BM25 index with {len(self.chunk_map)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading BM25 index: {e}")
            return False
    
    def _save_index(self):
        """Save BM25 index and mappings"""
        try:
            # Save BM25 retriever
            self.retriever.persist(str(self.persist_path))
            
            # Save mappings
            mappings_file = self.persist_path / "bm25_mappings.pkl"
            with open(mappings_file, 'wb') as f:
                pickle.dump({
                    'chunk_map': self.chunk_map,
                    'node_map': self.node_map
                }, f)
            
            logger.info(f"💾 BM25 index saved to {self.persist_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving BM25 index: {e}")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search using BM25 and return (chunk_id, score) pairs"""
        if not self.retriever:
            logger.warning("BM25 retriever not initialized")
            return []
        
        try:
            # Create query bundle
            query_bundle = QueryBundle(query_str=query)
            
            # Retrieve nodes
            nodes_with_scores = self.retriever.retrieve(query_bundle)
            
            # Convert to chunk results
            results = []
            seen_chunks = set()
            
            for node_with_score in nodes_with_scores[:k*2]:  # Get more for deduplication
                node_id = node_with_score.node.node_id
                chunk_id = self.node_map.get(node_id)
                
                if chunk_id and chunk_id not in seen_chunks:
                    results.append((chunk_id, node_with_score.score))
                    seen_chunks.add(chunk_id)
                    
                    if len(results) >= k:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"❌ BM25 search error: {e}")
            return []
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID"""
        return self.chunk_map.get(chunk_id)
    
    def analyze_index_quality(self) -> dict:
        """Analyze BM25 index quality metrics"""
        if not self.retriever:
            return {}
        
        stats = {
            'total_chunks': len(self.chunk_map),
            'total_nodes': len(self.node_map),
            'avg_nodes_per_chunk': len(self.node_map) / len(self.chunk_map) if self.chunk_map else 0,
            'chunk_types': {},
            'content_distribution': {},
            'metadata_coverage': {}
        }
        
        # Analyze chunk types
        for chunk in self.chunk_map.values():
            chunk_type = chunk.chunk_type
            stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
        
        # Analyze content length distribution
        content_lengths = [len(chunk.content) for chunk in self.chunk_map.values()]
        if content_lengths:
            stats['content_distribution'] = {
                'min_length': min(content_lengths),
                'max_length': max(content_lengths),
                'avg_length': sum(content_lengths) / len(content_lengths),
                'median_length': sorted(content_lengths)[len(content_lengths) // 2]
            }
        
        # Analyze metadata coverage
        metadata_keys = set()
        for chunk in self.chunk_map.values():
            metadata_keys.update(chunk.metadata.keys())
        
        stats['metadata_coverage'] = {
            'unique_metadata_keys': len(metadata_keys),
            'metadata_keys': list(metadata_keys)
        }
        
        return stats

    def suggest_optimizations(self) -> List[str]:
        """Suggest index optimizations based on analysis"""
        suggestions = []
        stats = self.analyze_index_quality()
        
        avg_nodes_per_chunk = stats.get('avg_nodes_per_chunk', 0)
        if avg_nodes_per_chunk > 3:
            suggestions.append("Consider larger chunk sizes to reduce node fragmentation")
        elif avg_nodes_per_chunk < 1.2:
            suggestions.append("Consider smaller chunk sizes for better granularity")
        
        content_dist = stats.get('content_distribution', {})
        if content_dist.get('max_length', 0) > 2000:
            suggestions.append("Some chunks are very long, consider better splitting")
        
        chunk_types = stats.get('chunk_types', {})
        if len(chunk_types) == 1:
            suggestions.append("Consider using different chunk types for better categorization")
        
        return suggestions

    def get_retrieval_stats(self) -> dict:
        """Get retrieval performance statistics"""
        if not self.retriever:
            return {}
        
        # Get BM25 internal stats (if available)
        stats = {
            'retriever_type': type(self.retriever).__name__,
            'similarity_top_k': getattr(self.retriever, 'similarity_top_k', None),
            'total_indexed_nodes': len(self.node_map),
            'unique_chunks': len(self.chunk_map)
        }
        
        return stats

    # Also improve the build_index method to be more robust:
    def build_index_robust(self, chunks: List[Chunk]) -> bool:
        """Build BM25 index from chunks with error recovery"""
        try:
            logger.info(f"Building BM25 index for {len(chunks)} chunks...")
            
            # Validate chunks first
            valid_chunks = []
            for chunk in chunks:
                if not chunk.content.strip():
                    logger.warning(f"Skipping empty chunk: {chunk.id}")
                    continue
                if len(chunk.content) > 10000:  # Very large chunks
                    logger.warning(f"Very large chunk detected: {chunk.id} ({len(chunk.content)} chars)")
                valid_chunks.append(chunk)
            
            logger.info(f"Processing {len(valid_chunks)} valid chunks out of {len(chunks)}")
            
            # Convert chunks to LlamaIndex nodes
            nodes = []
            self.chunk_map = {}
            self.node_map = {}
            failed_chunks = []
            
            for chunk in valid_chunks:
                try:
                    # Adaptive chunking based on chunk type
                    optimal_size = self._get_optimal_chunk_size(chunk)
                    splitter = SentenceSplitter(
                        chunk_size=optimal_size, 
                        chunk_overlap=optimal_size // 8
                    )

                    # Create LlamaIndex Document
                    doc = Document(
                        text=chunk.content,
                        metadata={
                            'chunk_id': chunk.id,
                            'chunk_type': chunk.chunk_type,
                            'document_title': chunk.metadata.get('document_title', ''),
                            'section_title': chunk.metadata.get('section_title', ''),
                            'hierarchy_path': chunk.metadata.get('hierarchy_path', ''),
                            'original_chunk_size': len(chunk.content),
                            'content_density': self._calculate_content_density(chunk.content)
                        }
                    )
                    
                    # Smart split content
                    chunk_nodes = self._smart_split_content(doc, splitter)
                    
                    for i, node in enumerate(chunk_nodes):
                        node_id = f"{chunk.id}_bm25_{i}"
                        node.node_id = node_id
                        
                        # Enhanced metadata for BM25
                        node.metadata.update({
                            'node_index': i,
                            'total_nodes': len(chunk_nodes),
                            'bm25_weight': self._calculate_bm25_weight(node.text, chunk)
                        })
                        
                        self.node_map[node_id] = chunk.id
                        self.chunk_map[chunk.id] = chunk
                        nodes.append(node)
                        
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk.id}: {e}")
                    failed_chunks.append(chunk.id)
                    continue
            
            if failed_chunks:
                logger.warning(f"Failed to process {len(failed_chunks)} chunks: {failed_chunks[:5]}...")
            
            if not nodes:
                logger.error("No valid nodes created for BM25 index")
                return False
            
            # Create BM25 retriever
            self.retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=15,  # Higher number for better fusion
                verbose=True
            )
            
            # Save index
            self._save_index()
            
            logger.info(f"✅ BM25 index built with {len(nodes)} nodes from {len(valid_chunks)} chunks")
            logger.info(f"📊 Index quality stats: {self.analyze_index_quality()}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building BM25 index: {e}")
            return False