# core/config.py - Fixed configuration with proper Qdrant support

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import os

@dataclass
class VectorStoreConfig:
    """Vector store configuration - supports both FAISS and Qdrant"""
    # Vector store type: 'faiss' or 'qdrant'
    store_type: str = "qdrant"
    
    # Common settings
    dimension: int = 768
    distance_metric: str = "Cosine"  # Cosine, Euclidean, Dot
    
    # FAISS specific settings
    faiss_index_type: str = "hnsw"  # hnsw, ivf, flat
    faiss_m: int = 32
    faiss_ef_construction: int = 200
    faiss_ef_search: int = 128
    
    # Qdrant specific settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "vietnamese_rag"
    qdrant_timeout: int = 60
    
    # HNSW configuration (used by both FAISS and Qdrant)
    hnsw_m: int = 32
    hnsw_ef_construct: int = 200
    hnsw_full_scan_threshold: int = 10000
    hnsw_max_indexing_threads: int = 0
    hnsw_on_disk: bool = False
    
    # Index optimization settings
    enable_quantization: bool = True
    quantization_type: str = "int8"  # int8, binary
    optimize_for_memory: bool = False

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "Alibaba-NLP/gte-multilingual-base"
    model_path: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    pooling_strategy: str = "mean"  # mean, cls, max
    model_dimension: int = 768  # Add model dimension for vector store initialization

@dataclass
class ChunkingConfig:
    """Text chunking configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 4096
    
    # Hierarchical chunking
    enable_hierarchical_chunking: bool = True
    parent_chunk_size: int = 2048
    child_chunk_size: int = 512
    overlap_tokens: int = 128
    
    # Table handling
    table_max_rows_per_chunk: int = 50
    table_include_headers: bool = True
    table_preserve_structure: bool = True

@dataclass
class TableConfig:
    """Table processing configuration"""
    max_table_tokens: int = 1024
    table_summary_tokens: int = 256
    preserve_structure: bool = True
    extract_headers: bool = True

@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    default_k: int = 5
    max_k: int = 20
    default_strategy: str = "enhanced_hierarchical"  # enhanced_hierarchical, hierarchical, table_aware, simple
    
    # Context formatting
    max_context_tokens: int = 4096
    include_parent_context: bool = True
    include_related_chunks: bool = True
    max_related_chunks: int = 3
    
    # Search thresholds
    min_similarity_threshold: float = 0.0
    rerank_results: bool = False

@dataclass
class ProcessingConfig:
    """Document processing configuration"""
    supported_formats: list = None
    extract_metadata: bool = True
    preserve_formatting: bool = True
    
    # Vietnamese specific
    enable_vietnamese_preprocessing: bool = True
    normalize_vietnamese_text: bool = True
    
    # Table processing
    extract_table_structure: bool = True
    table_extraction_method: str = "PyMuPDF"  # pdfplumber, camelot, tabula
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.docx', '.doc', '.txt', '.md']

@dataclass
class RAGConfig:
    """Main RAG system configuration"""
    # Sub-configurations
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    table: TableConfig = field(default_factory=TableConfig)
    
    # General settings
    index_path: str = "./indices/vietnamese_rag"
    metadata_path: str = "./qdrant_metadata"
    log_level: str = "INFO"
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_processing_size: int = 100
    
    # Development settings
    debug_mode: bool = False
    save_intermediate_results: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """Create config from dictionary"""
        vector_store_config = VectorStoreConfig(**config_dict.get('vector_store', {}))
        embedding_config = EmbeddingConfig(**config_dict.get('embedding', {}))
        chunking_config = ChunkingConfig(**config_dict.get('chunking', {}))
        retrieval_config = RetrievalConfig(**config_dict.get('retrieval', {}))
        processing_config = ProcessingConfig(**config_dict.get('processing', {}))
        table_config = TableConfig(**config_dict.get('table', {}))
        
        # Get main config excluding sub-configs
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['vector_store', 'embedding', 'chunking', 'retrieval', 'processing', 'table']}
        
        return cls(
            vector_store=vector_store_config,
            embedding=embedding_config,
            chunking=chunking_config,
            retrieval=retrieval_config,
            processing=processing_config,
            table=table_config,
            **main_config
        )
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create config from environment variables"""
        config = cls()
        
        # Vector store settings
        config.vector_store.store_type = os.getenv('RAG_VECTOR_STORE_TYPE', config.vector_store.store_type)
        config.vector_store.qdrant_host = os.getenv('RAG_QDRANT_HOST', config.vector_store.qdrant_host)
        config.vector_store.qdrant_port = int(os.getenv('RAG_QDRANT_PORT', config.vector_store.qdrant_port))
        config.vector_store.qdrant_collection_name = os.getenv('RAG_QDRANT_COLLECTION', config.vector_store.qdrant_collection_name)
        
        # Embedding settings
        config.embedding.model_name = os.getenv('RAG_EMBEDDING_MODEL', config.embedding.model_name)
        config.embedding.device = os.getenv('RAG_DEVICE', config.embedding.device)
        
        # General settings
        config.index_path = os.getenv('RAG_INDEX_PATH', config.index_path)
        config.metadata_path = os.getenv('RAG_METADATA_PATH', config.metadata_path)
        config.log_level = os.getenv('RAG_LOG_LEVEL', config.log_level)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'vector_store': {
                'store_type': self.vector_store.store_type,
                'dimension': self.vector_store.dimension,
                'distance_metric': self.vector_store.distance_metric,
                'faiss_index_type': self.vector_store.faiss_index_type,
                'faiss_m': self.vector_store.faiss_m,
                'faiss_ef_construction': self.vector_store.faiss_ef_construction,
                'faiss_ef_search': self.vector_store.faiss_ef_search,
                'qdrant_host': self.vector_store.qdrant_host,
                'qdrant_port': self.vector_store.qdrant_port,
                'qdrant_collection_name': self.vector_store.qdrant_collection_name,
                'qdrant_timeout': self.vector_store.qdrant_timeout,
                'hnsw_m': self.vector_store.hnsw_m,
                'hnsw_ef_construct': self.vector_store.hnsw_ef_construct,
                'hnsw_full_scan_threshold': self.vector_store.hnsw_full_scan_threshold,
                'hnsw_max_indexing_threads': self.vector_store.hnsw_max_indexing_threads,
                'hnsw_on_disk': self.vector_store.hnsw_on_disk,
                'enable_quantization': self.vector_store.enable_quantization,
                'quantization_type': self.vector_store.quantization_type,
                'optimize_for_memory': self.vector_store.optimize_for_memory
            },
            'embedding': {
                'model_name': self.embedding.model_name,
                'model_path': self.embedding.model_path,
                'device': self.embedding.device,
                'batch_size': self.embedding.batch_size,
                'max_length': self.embedding.max_length,
                'normalize_embeddings': self.embedding.normalize_embeddings,
                'pooling_strategy': self.embedding.pooling_strategy,
                'model_dimension': self.embedding.model_dimension
            },
            'chunking': {
                'chunk_size': self.chunking.chunk_size,
                'chunk_overlap': self.chunking.chunk_overlap,
                'min_chunk_size': self.chunking.min_chunk_size,
                'max_chunk_size': self.chunking.max_chunk_size,
                'enable_hierarchical_chunking': self.chunking.enable_hierarchical_chunking,
                'parent_chunk_size': self.chunking.parent_chunk_size,
                'child_chunk_size': self.chunking.child_chunk_size,
                'overlap_tokens': self.chunking.overlap_tokens,
                'table_max_rows_per_chunk': self.chunking.table_max_rows_per_chunk,
                'table_include_headers': self.chunking.table_include_headers,
                'table_preserve_structure': self.chunking.table_preserve_structure
            },
            'retrieval': {
                'default_k': self.retrieval.default_k,
                'max_k': self.retrieval.max_k,
                'default_strategy': self.retrieval.default_strategy,
                'max_context_tokens': self.retrieval.max_context_tokens,
                'include_parent_context': self.retrieval.include_parent_context,
                'include_related_chunks': self.retrieval.include_related_chunks,
                'max_related_chunks': self.retrieval.max_related_chunks,
                'min_similarity_threshold': self.retrieval.min_similarity_threshold,
                'rerank_results': self.retrieval.rerank_results
            },
            'processing': {
                'supported_formats': self.processing.supported_formats,
                'extract_metadata': self.processing.extract_metadata,
                'preserve_formatting': self.processing.preserve_formatting,
                'enable_vietnamese_preprocessing': self.processing.enable_vietnamese_preprocessing,
                'normalize_vietnamese_text': self.processing.normalize_vietnamese_text,
                'extract_table_structure': self.processing.extract_table_structure,
                'table_extraction_method': self.processing.table_extraction_method
            },
            'table': {
                'max_table_tokens': self.table.max_table_tokens,
                'table_summary_tokens': self.table.table_summary_tokens,
                'preserve_structure': self.table.preserve_structure,
                'extract_headers': self.table.extract_headers
            },
            'index_path': self.index_path,
            'metadata_path': self.metadata_path,
            'log_level': self.log_level,
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_workers': self.max_workers,
            'batch_processing_size': self.batch_processing_size,
            'debug_mode': self.debug_mode,
            'save_intermediate_results': self.save_intermediate_results
        }

# Preset configurations for different use cases
class ConfigPresets:
    """Predefined configuration presets"""
    
    @staticmethod
    def development() -> RAGConfig:
        """Configuration for development environment"""
        config = RAGConfig()
        config.vector_store.store_type = "qdrant"
        config.vector_store.qdrant_collection_name = "vietnamese_rag_dev"
        config.embedding.batch_size = 16
        config.chunking.chunk_size = 800
        config.retrieval.default_k = 3
        config.debug_mode = True
        config.save_intermediate_results = True
        return config
    
    @staticmethod
    def production() -> RAGConfig:
        """Configuration for production environment"""
        config = RAGConfig()
        config.vector_store.store_type = "qdrant"
        config.vector_store.qdrant_collection_name = "vietnamese_rag_prod"
        config.vector_store.enable_quantization = True
        config.vector_store.optimize_for_memory = True
        config.embedding.batch_size = 64
        config.chunking.chunk_size = 1000
        config.retrieval.default_k = 5
        config.enable_parallel_processing = True
        config.max_workers = 8
        config.debug_mode = False
        return config
    
    @staticmethod
    def high_memory() -> RAGConfig:
        """Configuration for high memory systems"""
        config = RAGConfig()
        config.vector_store.store_type = "qdrant"
        config.vector_store.hnsw_m = 64
        config.vector_store.hnsw_ef_construct = 400
        config.vector_store.enable_quantization = False
        config.embedding.batch_size = 128
        config.chunking.chunk_size = 1500
        config.retrieval.default_k = 10
        config.max_workers = 16
        return config
    
    @staticmethod
    def low_memory() -> RAGConfig:
        """Configuration for low memory systems"""
        config = RAGConfig()
        config.vector_store.store_type = "qdrant"
        config.vector_store.hnsw_m = 16
        config.vector_store.hnsw_ef_construct = 100
        config.vector_store.enable_quantization = True
        config.vector_store.optimize_for_memory = True
        config.embedding.batch_size = 8
        config.chunking.chunk_size = 600
        config.retrieval.default_k = 3
        config.max_workers = 2
        return config
    
    @staticmethod
    def table_focused() -> RAGConfig:
        """Configuration optimized for table processing"""
        config = RAGConfig()
        config.vector_store.store_type = "qdrant"
        config.chunking.table_max_rows_per_chunk = 30
        config.chunking.table_include_headers = True
        config.chunking.table_preserve_structure = True
        config.processing.extract_table_structure = True
        config.retrieval.default_strategy = "table_aware"
        return config

# Factory function to create vector store based on config
def create_vector_store(config: RAGConfig):
    """Factory function to create vector store based on configuration"""
    if config.vector_store.store_type == "qdrant":
        from retrieval.vector_db_qdrant import QdrantVectorStore
        
        hnsw_config = {
            "m": config.vector_store.hnsw_m,
            "ef_construct": config.vector_store.hnsw_ef_construct,
            "full_scan_threshold": config.vector_store.hnsw_full_scan_threshold,
            "max_indexing_threads": config.vector_store.hnsw_max_indexing_threads,
            "on_disk": config.vector_store.hnsw_on_disk
        }
        
        return QdrantVectorStore(
            collection_name=config.vector_store.qdrant_collection_name,
            host=config.vector_store.qdrant_host,
            port=config.vector_store.qdrant_port,
            dimension=config.vector_store.dimension,
            distance_metric=config.vector_store.distance_metric,
            hnsw_m=config.vector_store.hnsw_m,
            hnsw_ef_construct=config.vector_store.hnsw_ef_construct,
            hnsw_ef_search=config.vector_store.faiss_ef_search
        )
    
    elif config.vector_store.store_type == "faiss":
        # Import FAISS vector store if needed
        try:
            from retrieval.retriever import FAISSVectorStore
            return FAISSVectorStore(
                dimension=config.vector_store.dimension,
                index_type=config.vector_store.faiss_index_type,
                m=config.vector_store.faiss_m,
                ef_construction=config.vector_store.faiss_ef_construction,
                ef_search=config.vector_store.faiss_ef_search,
                metric=config.vector_store.distance_metric.lower()
            )
        except ImportError:
            raise ImportError("FAISS vector store not available. Please use Qdrant instead.")
    
    else:
        raise ValueError(f"Unsupported vector store type: {config.vector_store.store_type}")

# Example usage and testing
if __name__ == "__main__":
    # Test different configurations
    print("=== Development Config ===")
    dev_config = ConfigPresets.development()
    print(f"Vector Store: {dev_config.vector_store.store_type}")
    print(f"Collection: {dev_config.vector_store.qdrant_collection_name}")
    print(f"Chunk Size: {dev_config.chunking.chunk_size}")
    
    print("\n=== Production Config ===")
    prod_config = ConfigPresets.production()
    print(f"Vector Store: {prod_config.vector_store.store_type}")
    print(f"Quantization: {prod_config.vector_store.enable_quantization}")
    print(f"Batch Size: {prod_config.embedding.batch_size}")
    
    print("\n=== Environment Config ===")
    env_config = RAGConfig.from_env()
    print(f"Qdrant Host: {env_config.vector_store.qdrant_host}")
    print(f"Qdrant Port: {env_config.vector_store.qdrant_port}")
    
    # Test configuration serialization
    print("\n=== Config Serialization Test ===")
    config_dict = dev_config.to_dict()
    restored_config = RAGConfig.from_dict(config_dict)
    print(f"Original chunk size: {dev_config.chunking.chunk_size}")
    print(f"Restored chunk size: {restored_config.chunking.chunk_size}")
    print(f"Serialization test: {'PASSED' if dev_config.chunking.chunk_size == restored_config.chunking.chunk_size else 'FAILED'}")
    
    # Test vector store factory
    print("\n=== Vector Store Factory Test ===")
    try:
        vector_store = create_vector_store(dev_config)
        print(f"Vector store created successfully: {type(vector_store).__name__}")
    except Exception as e:
        print(f"Vector store creation failed: {e}")
        print("Make sure Qdrant is running on localhost:6333")