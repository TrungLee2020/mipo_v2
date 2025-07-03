from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import pickle
import os
from abc import ABC, abstractmethod
from processing.chunker import Chunk
from processing.embedder import VietnameseEmbedder
from utils.utils import estimate_tokens
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

class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store with UUID point IDs"""
    
    def __init__(self, 
                 collection_name: str = "vietnamese_rag",
                 host: str = "localhost",
                 port: int = 6333,
                 dimension: int = 768,
                 distance_metric: str = "Cosine",
                 hnsw_m: int = 32,
                 hnsw_ef_construct: int = 200,
                 hnsw_ef_search: int = 128):
        """Initialize Qdrant vector store with proper UUID handling"""
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.dimension = dimension
        self.distance_metric = distance_metric
        
        # HNSW configuration
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.hnsw_ef_search = hnsw_ef_search
        
        # Initialize Qdrant client
        try:
            self.client = QdrantClient(host=self.host, port=self.port, timeout=60)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
        
        # ID mappings: chunk_id <-> qdrant_point_id
        self.chunk_to_point_id = {}  # chunk_id -> UUID
        self.point_to_chunk_id = {}  # UUID -> chunk_id
        
        # Local mappings for enhanced functionality
        self.chunk_map = {}  # chunk_id -> Chunk
        self.parent_to_children = {}  # parent_id -> [child_ids]
        self.child_to_parent = {}     # child_id -> parent_id
        
        # Initialize collection
        self._initialize_collection()
        
        logger.info(f"Initialized Qdrant collection '{collection_name}' with dimension {dimension}")
    
    def _initialize_collection(self):
        """Initialize Qdrant collection with proper configuration"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if not collection_exists:
                # Map distance metric
                distance_mapping = {
                    "Cosine": Distance.COSINE,
                    "cosine": Distance.COSINE,
                    "Euclidean": Distance.EUCLID,
                    "l2": Distance.EUCLID,
                    "Dot": Distance.DOT,
                    "ip": Distance.DOT
                }
                distance = distance_mapping.get(self.distance_metric, Distance.COSINE)
                
                # Create collection with HNSW configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=distance,
                        on_disk=True  # Store vectors on disk for better memory usage
                    ),
                    hnsw_config=HnswConfigDiff(
                        m=self.hnsw_m,
                        ef_construct=self.hnsw_ef_construct,
                        full_scan_threshold=10000,
                        max_indexing_threads=0,
                        on_disk=False
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        deleted_threshold=0.2,
                        vacuum_min_vector_number=1000,
                        default_segment_number=0,
                        max_segment_size=None,
                        memmap_threshold=None,
                        indexing_threshold=20000,
                        flush_interval_sec=5,
                        max_optimization_threads=1
                    )
                )
                logger.info(f"Created collection '{self.collection_name}' with HNSW index")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")
                # Load existing chunks and mappings
                self._load_existing_chunks()
                
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise
    
    def _load_existing_chunks(self):
        """Load existing chunks from Qdrant into local mappings"""
        try:
            # Get all points with scroll
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            for point in points:
                point_id = point.id
                chunk_data = point.payload.get('chunk_data')
                original_chunk_id = point.payload.get('original_chunk_id')
                
                if chunk_data and original_chunk_id:
                    # Deserialize chunk
                    try:
                        chunk = pickle.loads(bytes.fromhex(chunk_data))
                        
                        # Update mappings
                        self.chunk_to_point_id[original_chunk_id] = point_id
                        self.point_to_chunk_id[point_id] = original_chunk_id
                        self.chunk_map[original_chunk_id] = chunk
                        
                        # Update parent-child mappings
                        self._update_parent_child_mappings(chunk)
                        
                    except Exception as e:
                        logger.warning(f"Could not deserialize chunk {point_id}: {e}")
            
            logger.info(f"Loaded {len(self.chunk_map)} existing chunks")
            
        except Exception as e:
            logger.warning(f"Could not load existing chunks: {e}")
    
    def _generate_point_id(self, chunk_id: str) -> str:
        """Generate a valid UUID for Qdrant point ID"""
        # Check if we already have a mapping
        if chunk_id in self.chunk_to_point_id:
            return self.chunk_to_point_id[chunk_id]
        
        # Generate a deterministic UUID based on chunk_id
        # This ensures the same chunk_id always gets the same UUID
        namespace = uuid.UUID('12345678-1234-5678-1234-123456789abc')
        point_id = str(uuid.uuid5(namespace, chunk_id))
        
        # Store the mapping
        self.chunk_to_point_id[chunk_id] = point_id
        self.point_to_chunk_id[point_id] = chunk_id
        
        return point_id
    
    def add_embeddings(self, embeddings: Dict[str, np.ndarray], chunks: List[Chunk]):
        """Add embeddings to Qdrant with proper UUID point IDs"""
        chunk_dict = {chunk.id: chunk for chunk in chunks}
        
        points = []
        
        for chunk_id, embedding in embeddings.items():
            if chunk_id in chunk_dict and chunk_id not in self.chunk_map:
                chunk = chunk_dict[chunk_id]
                
                # Generate valid point ID (UUID)
                point_id = self._generate_point_id(chunk_id)
                
                # Normalize embedding if using cosine distance
                if self.distance_metric.lower() in ["cosine"]:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                
                # Prepare payload with comprehensive metadata
                payload = self._create_payload(chunk, chunk_id)
                
                # Create point with UUID
                point = PointStruct(
                    id=point_id,  # Use UUID instead of chunk_id
                    vector=embedding.tolist(),
                    payload=payload
                )
                points.append(point)
                
                # Update local mappings
                self.chunk_map[chunk_id] = chunk
                self._update_parent_child_mappings(chunk)
        
        if points:
            try:
                # Batch upload points with retry logic
                max_retries = 3
                retry_delay = 2
                
                for attempt in range(max_retries):
                    try:
                        self.client.upsert(
                            collection_name=self.collection_name,
                            points=points,
                            wait=True
                        )
                        logger.info(f"Added {len(points)} embeddings to Qdrant collection")
                        break
                    except Exception as e:
                        logger.error(f"Error upserting points, attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            import time
                            time.sleep(retry_delay)
                        else:
                            raise
                
            except Exception as e:
                logger.error(f"Error adding embeddings to Qdrant: {e}")
                raise
    
    def _create_payload(self, chunk: Chunk, original_chunk_id: str) -> Dict:
        """Create comprehensive payload for Qdrant point"""
        metadata = chunk.metadata or {}
        
        payload = {
            # Store original chunk ID and serialized chunk data
            'original_chunk_id': original_chunk_id,
            'chunk_data': pickle.dumps(chunk).hex(),
            
            # Basic chunk information
            'chunk_type': chunk.chunk_type,
            'chunk_level': metadata.get('chunk_level', 'unknown'),
            'content_length': len(chunk.content),
            'content_preview': chunk.content[:500],
            
            # Document metadata
            'document_title': metadata.get('document_title', ''),
            'document_id': metadata.get('document_id', ''),
            'file_name': metadata.get('file_name', ''),
            'file_path': metadata.get('file_path', ''),
            
            # Hierarchical information
            'section_title': metadata.get('section_title', ''),
            'section_level': metadata.get('section_level', 0),
            'hierarchy_path': metadata.get('hierarchy_path', ''),
            
            # Parent-child relationships
            'parent_id': chunk.parent_id or '',
            'child_ids': chunk.child_ids or [],
            
            # Administrative structure
            'admin_section': metadata.get('administrative_info', {}).get('section', ''),
            'admin_chapter': metadata.get('administrative_info', {}).get('chapter', ''),
            'admin_article': metadata.get('administrative_info', {}).get('article', ''),
            'admin_point': metadata.get('administrative_info', {}).get('point', ''),
            
            # Content type and processing info
            'content_type': metadata.get('content_type', 'text'),
            'processing_timestamp': datetime.now().isoformat(),
            
            # Line information
            'start_line': metadata.get('start_line', 0),
            'end_line': metadata.get('end_line', 0),
            
            # Table information (if applicable)
            'is_table_content': 'table' in chunk.chunk_type,
            'table_row_count': metadata.get('table_info', {}).get('row_count', 0),
            'table_column_count': metadata.get('table_info', {}).get('column_count', 0),
            
            # Enhanced metadata
            'has_parent': bool(chunk.parent_id),
            'has_children': bool(chunk.child_ids),
            'breadcrumb_titles': [item.get('title', '') for item in metadata.get('breadcrumb', [])],
        }
        
        # Add table headers if available
        table_info = metadata.get('table_info', {})
        if table_info.get('headers'):
            payload['table_headers'] = table_info['headers'][:10]
        
        return payload
    
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
        """Search for similar chunks using Qdrant with UUID point IDs"""
        # Normalize query embedding if using cosine distance
        if self.distance_metric.lower() in ["cosine"]:
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        # Build filter if provided
        search_filter = self._build_filter(filter_dict) if filter_dict else None
        
        # Search with more candidates for context enrichment
        search_k = min(k * 5, 1000)
        
        try:
            # Perform HNSW search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=search_k,
                query_filter=search_filter,
                with_payload=True,
                score_threshold=0.0
            )
            
            results = []
            processed_chunks = set()
            
            for scored_point in search_result:
                point_id = scored_point.id
                score = scored_point.score
                
                # Get original chunk ID
                original_chunk_id = self.point_to_chunk_id.get(point_id)
                if not original_chunk_id:
                    original_chunk_id = scored_point.payload.get('original_chunk_id')
                    if original_chunk_id:
                        self.point_to_chunk_id[point_id] = original_chunk_id
                        self.chunk_to_point_id[original_chunk_id] = point_id
                
                if not original_chunk_id or original_chunk_id in processed_chunks:
                    continue
                
                # Get chunk from local map or reconstruct from payload
                chunk = self._get_chunk_from_result(scored_point, original_chunk_id)
                if not chunk:
                    continue
                
                # Create enhanced result with parent-child context
                enhanced_result = self._create_enhanced_result(chunk, score, original_chunk_id)
                
                if enhanced_result:
                    results.append(enhanced_result)
                    processed_chunks.add(original_chunk_id)
                    
                    # Mark related chunks as processed to avoid duplicates
                    if enhanced_result.parent_chunk:
                        processed_chunks.add(enhanced_result.parent_chunk.id)
                    for related_chunk in enhanced_result.related_chunks:
                        processed_chunks.add(related_chunk.id)
                    
                    if len(results) >= k:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching in Qdrant: {e}")
            return []
    
    def _build_filter(self, filter_dict: Dict) -> Filter:
        """Build Qdrant filter from filter dictionary"""
        conditions = []
        
        for key, value in filter_dict.items():
            if key == 'chunk_type':
                if isinstance(value, list):
                    # Multiple chunk types
                    chunk_conditions = [
                        FieldCondition(key="chunk_type", match=MatchValue(value=v))
                        for v in value
                    ]
                    if len(chunk_conditions) == 1:
                        conditions.append(chunk_conditions[0])
                    else:
                        conditions.append(models.Filter(should=chunk_conditions))
                else:
                    conditions.append(
                        FieldCondition(key="chunk_type", match=MatchValue(value=value))
                    )
            
            elif key == 'document_title':
                conditions.append(
                    FieldCondition(key="document_title", match=MatchValue(value=value))
                )
            
            elif key == 'content_type':
                conditions.append(
                    FieldCondition(key="content_type", match=MatchValue(value=value))
                )
            
            elif key == 'has_parent':
                conditions.append(
                    FieldCondition(key="has_parent", match=MatchValue(value=value))
                )
            
            elif key == 'is_table_content':
                conditions.append(
                    FieldCondition(key="is_table_content", match=MatchValue(value=value))
                )
        
        if conditions:
            if len(conditions) == 1:
                return Filter(must=conditions)
            else:
                return Filter(must=conditions)
        
        return None
    
    def _get_chunk_from_result(self, scored_point, original_chunk_id: str) -> Optional[Chunk]:
        """Get chunk from search result using original chunk ID"""
        # Try local map first
        if original_chunk_id in self.chunk_map:
            return self.chunk_map[original_chunk_id]
        
        # Reconstruct from payload
        try:
            chunk_data = scored_point.payload.get('chunk_data')
            if chunk_data:
                chunk = pickle.loads(bytes.fromhex(chunk_data))
                self.chunk_map[original_chunk_id] = chunk  # Cache it
                self._update_parent_child_mappings(chunk)
                return chunk
        except Exception as e:
            logger.warning(f"Could not reconstruct chunk {original_chunk_id}: {e}")
        
        return None
    
    def _create_enhanced_result(self, chunk: Chunk, score: float, chunk_id: str) -> Optional[RetrievalResult]:
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
                if sibling_id != chunk_id and sibling_id in self.chunk_map:
                    related_chunks.append(self.chunk_map[sibling_id])
            
            context_metadata['sibling_count'] = len(related_chunks)
            
        elif chunk.chunk_type in ['parent', 'table_parent']:
            # Get child chunks
            child_ids = self.parent_to_children.get(chunk_id, [])
            for child_id in child_ids:
                if child_id in self.chunk_map:
                    related_chunks.append(self.chunk_map[child_id])
            
            context_metadata['child_count'] = len(related_chunks)
        
        # Add hierarchy context metadata
        context_metadata.update(self._extract_hierarchy_context(chunk, parent_chunk))
        
        return RetrievalResult(
            chunk_id=chunk_id,
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
        """Get chunk by original chunk ID"""
        # Try local map first
        if chunk_id in self.chunk_map:
            return self.chunk_map[chunk_id]
        
        # Query Qdrant using original chunk ID
        try:
            # Search by original_chunk_id in payload
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="original_chunk_id", match=MatchValue(value=chunk_id))]
                ),
                limit=1,
                with_payload=True
            )
            
            points, _ = search_result
            if points:
                point = points[0]
                chunk_data = point.payload.get('chunk_data')
                if chunk_data:
                    chunk = pickle.loads(bytes.fromhex(chunk_data))
                    self.chunk_map[chunk_id] = chunk  # Cache it
                    
                    # Update ID mappings
                    self.chunk_to_point_id[chunk_id] = point.id
                    self.point_to_chunk_id[point.id] = chunk_id
                    
                    return chunk
        except Exception as e:
            logger.warning(f"Could not retrieve chunk {chunk_id}: {e}")
        
        return None
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, Chunk]:
        """Get multiple chunks by original chunk IDs"""
        result = {}
        missing_ids = []
        
        # Check local map first
        for chunk_id in chunk_ids:
            if chunk_id in self.chunk_map:
                result[chunk_id] = self.chunk_map[chunk_id]
            else:
                missing_ids.append(chunk_id)
        
        # Query Qdrant for missing chunks
        if missing_ids:
            try:
                # Search by original_chunk_id in payload
                search_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        should=[
                            FieldCondition(key="original_chunk_id", match=MatchValue(value=chunk_id))
                            for chunk_id in missing_ids
                        ]
                    ),
                    limit=len(missing_ids),
                    with_payload=True
                )
                
                points, _ = search_result
                for point in points:
                    original_chunk_id = point.payload.get('original_chunk_id')
                    chunk_data = point.payload.get('chunk_data')
                    
                    if original_chunk_id and chunk_data:
                        chunk = pickle.loads(bytes.fromhex(chunk_data))
                        self.chunk_map[original_chunk_id] = chunk  # Cache it
                        result[original_chunk_id] = chunk
                        
                        # Update ID mappings
                        self.chunk_to_point_id[original_chunk_id] = point.id
                        self.point_to_chunk_id[point.id] = original_chunk_id
                        
            except Exception as e:
                logger.warning(f"Could not retrieve chunks {missing_ids}: {e}")
        
        return result
    
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks from Qdrant using original chunk IDs"""
        try:
            # Convert chunk IDs to point IDs
            point_ids = []
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_to_point_id:
                    point_ids.append(self.chunk_to_point_id[chunk_id])
                else:
                    # Try to find point ID by searching
                    try:
                        search_result = self.client.scroll(
                            collection_name=self.collection_name,
                            scroll_filter=Filter(
                                must=[FieldCondition(key="original_chunk_id", match=MatchValue(value=chunk_id))]
                            ),
                            limit=1,
                            with_payload=False
                        )
                        points, _ = search_result
                        if points:
                            point_ids.append(points[0].id)
                    except Exception as e:
                        logger.warning(f"Could not find point ID for chunk {chunk_id}: {e}")
            
            if point_ids:
                # Delete from Qdrant
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
            
            # Clean up local mappings
            deleted_count = 0
            for chunk_id in chunk_ids:
                if chunk_id in self.chunk_map:
                    self._cleanup_parent_child_mappings(chunk_id)
                    del self.chunk_map[chunk_id]
                    
                    # Clean up ID mappings
                    if chunk_id in self.chunk_to_point_id:
                        point_id = self.chunk_to_point_id[chunk_id]
                        del self.chunk_to_point_id[chunk_id]
                        if point_id in self.point_to_chunk_id:
                            del self.point_to_chunk_id[point_id]
                    
                    deleted_count += 1
            
            logger.info(f"Deleted {deleted_count} chunks from Qdrant")
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
        """Rebuild index to reclaim space"""
        logger.info("Optimizing Qdrant collection...")
        
        try:
            # Optimize collection
            self.client.update_collection(
                collection_name=self.collection_name,
                optimizer_config=models.OptimizersConfigDiff(
                    deleted_threshold=0.1,
                    vacuum_min_vector_number=1000,
                    max_optimization_threads=2
                )
            )
            logger.info("Qdrant collection optimized successfully")
            
        except Exception as e:
            logger.error(f"Error optimizing collection: {e}")
    
    def save_index(self, path: str) -> bool:
        """Save collection info and local mappings to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save metadata and mappings
            metadata = {
                'collection_name': self.collection_name,
                'host': self.host,
                'port': self.port,
                'dimension': self.dimension,
                'distance_metric': self.distance_metric,
                'hnsw_m': self.hnsw_m,
                'hnsw_ef_construct': self.hnsw_ef_construct,
                'hnsw_ef_search': self.hnsw_ef_search,
                'chunk_to_point_id': self.chunk_to_point_id,
                'point_to_chunk_id': self.point_to_chunk_id,
                'parent_to_children': self.parent_to_children,
                'child_to_parent': self.child_to_parent,
                'saved_timestamp': datetime.now().isoformat()
            }
            
            with open(f"{path}.qdrant_metadata", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved Qdrant metadata to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving Qdrant metadata: {e}")
            return False
    
    def load_index(self, path: str) -> bool:
        """Load collection info and local mappings from disk"""
        try:
            metadata_file = f"{path}.qdrant_metadata"
            if not os.path.exists(metadata_file):
                logger.info(f"No metadata file found at {metadata_file}")
                return False
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Restore configuration
            self.collection_name = metadata['collection_name']
            self.host = metadata['host']
            self.port = metadata['port']
            self.dimension = metadata['dimension']
            self.distance_metric = metadata['distance_metric']
            self.hnsw_m = metadata['hnsw_m']
            self.hnsw_ef_construct = metadata['hnsw_ef_construct']
            self.hnsw_ef_search = metadata['hnsw_ef_search']
            
            # Restore mappings
            self.chunk_to_point_id = metadata.get('chunk_to_point_id', {})
            self.point_to_chunk_id = metadata.get('point_to_chunk_id', {})
            self.parent_to_children = metadata.get('parent_to_children', {})
            self.child_to_parent = metadata.get('child_to_parent', {})
            
            # Reconnect to Qdrant and reload chunks
            self.client = QdrantClient(host=self.host, port=self.port, timeout=60)
            self._load_existing_chunks()
            
            logger.info(f"Loaded Qdrant metadata from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Qdrant metadata: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'total_vectors': collection_info.vectors_count,
                'collection_name': self.collection_name,
                'dimension': self.dimension,
                'distance_metric': self.distance_metric,
                'index_type': 'HNSW',
                'active_chunks': len(self.chunk_map),
                'parent_child_relationships': len(self.parent_to_children),
                'child_chunks': len(self.child_to_parent),
                'collection_status': collection_info.status,
                'optimizer_status': str(collection_info.optimizer_status),
                'hnsw_m': self.hnsw_m,
                'hnsw_ef_construct': self.hnsw_ef_construct,
                'hnsw_ef_search': self.hnsw_ef_search,
                'chunk_id_mappings': len(self.chunk_to_point_id)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'error': str(e),
                'active_chunks': len(self.chunk_map),
                'parent_child_relationships': len(self.parent_to_children),
                'child_chunks': len(self.child_to_parent),
                'chunk_id_mappings': len(self.chunk_to_point_id)
            }