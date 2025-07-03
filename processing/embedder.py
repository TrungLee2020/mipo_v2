
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from core.config import EmbeddingConfig
from processing.chunker import Chunk
from utils.utils import VietnameseTextProcessor
import re
import logging

logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class VietnameseEmbedder:
    """Vietnamese-optimized embedding for RAG"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._load_model()
    
    def _load_model(self):
        """Load Vietnamese embedding model"""
        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.device,
            trust_remote_code=True,
        )
        logger.info(f"Loaded model {self.config.model_name} on {self.device}")
    
    def embed_chunks(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """Embed list of chunks"""
        embeddings = {}
        
        # Group chunks by type for optimized processing
        text_chunks = [c for c in chunks if c.chunk_type in ['parent', 'child']]
        table_chunks = [c for c in chunks if 'table' in c.chunk_type]
        
        # Process text chunks
        if text_chunks:
            text_embeddings = self._embed_text_chunks(text_chunks)
            embeddings.update(text_embeddings)
        
        # Process table chunks
        if table_chunks:
            table_embeddings = self._embed_table_chunks(table_chunks)
            embeddings.update(table_embeddings)
        
        return embeddings
    
    def _embed_text_chunks(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """Embed text chunks with batching"""
        embeddings = {}
        
        # Prepare texts
        texts = []
        chunk_ids = []
        
        for chunk in chunks:
            # Preprocess text for embedding
            processed_text = self._preprocess_for_embedding(chunk.content)
            texts.append(processed_text)
            chunk_ids.append(chunk.id)
        
        # Batch embedding
        try:
            batch_embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Map back to chunk IDs
            for chunk_id, embedding in zip(chunk_ids, batch_embeddings):
                embeddings[chunk_id] = embedding
                
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            # Fallback to individual embedding
            for chunk_id, text in zip(chunk_ids, texts):
                try:
                    embedding = self.model.encode([text])[0]
                    embeddings[chunk_id] = embedding
                except Exception as e2:
                    logger.error(f"Failed to embed chunk {chunk_id}: {e2}")
        
        return embeddings
    
    def _embed_table_chunks(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """Embed table chunks with special handling"""
        embeddings = {}
        
        for chunk in chunks:
            try:
                if chunk.chunk_type == 'table_child':
                    # Special processing for table content
                    processed_text = self._preprocess_table_for_embedding(chunk)
                else:
                    processed_text = self._preprocess_for_embedding(chunk.content)
                
                embedding = self.model.encode([processed_text])[0]
                embeddings[chunk.id] = embedding
                
            except Exception as e:
                logger.error(f"Failed to embed table chunk {chunk.id}: {e}")
        
        return embeddings
    
    def _preprocess_for_embedding(self, text: str) -> str:
        """Preprocess Vietnamese text for embedding"""
        # Normalize
        text = VietnameseTextProcessor.normalize_diacritics(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
        text = re.sub(r'#{1,6}\s+', '', text)         # Remove headers
        
        # Truncate if too long
        words = text.split()
        if len(words) > self.config.max_length:
            text = ' '.join(words[:self.config.max_length])
        
        return text.strip()
    
    def _preprocess_table_for_embedding(self, chunk: Chunk) -> str:
        """Special preprocessing for table chunks"""
        content = chunk.content
        
        # Extract table info from metadata
        table_info = chunk.metadata.get('table_info', {})
        
        # Create structured representation
        if table_info:
            structured_text = []
            
            # Add table description
            headers = table_info.get('headers', [])
            if headers:
                structured_text.append(f"Bảng với các cột: {', '.join(headers)}")
            
            # Add row count info
            row_count = table_info.get('row_count', 0)
            if row_count:
                structured_text.append(f"Có {row_count} hàng dữ liệu")
            
            # Add actual content
            structured_text.append(content)
            
            return ' | '.join(structured_text)
        
        return self._preprocess_for_embedding(content)
    
    async def embed_chunks_async(self, chunks: List[Chunk]) -> Dict[str, np.ndarray]:
        """Async embedding for large batches"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Split chunks into batches
            batch_size = 100
            chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
            
            # Process batches in parallel
            tasks = []
            for batch in chunk_batches:
                task = loop.run_in_executor(executor, self.embed_chunks, batch)
                tasks.append(task)
            
            # Combine results
            batch_results = await asyncio.gather(*tasks)
            
            combined_embeddings = {}
            for result in batch_results:
                combined_embeddings.update(result)
            
            return combined_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed search query"""
        processed_query = self._preprocess_for_embedding(query)
        return self.model.encode([processed_query])[0]