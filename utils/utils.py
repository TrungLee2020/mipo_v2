# utils/utils.py - Updated with better chunk ID generation for Qdrant

import re
import unicodedata
import hashlib
import uuid
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VietnameseTextProcessor:
    """Vietnamese text processing utilities"""
    
    @staticmethod
    def normalize_diacritics(text: str) -> str:
        """Normalize Vietnamese diacritics"""
        # Convert to NFC form (canonical decomposition followed by canonical composition)
        normalized = unicodedata.normalize('NFC', text)
        return normalized
    
    @staticmethod
    def extract_vietnamese_sentences(text: str) -> List[str]:
        """Split Vietnamese text into sentences"""
        # Vietnamese sentence boundaries
        sentence_endings = r'[.!?;]\s+'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def detect_administrative_structure(text: str) -> Dict[str, List[str]]:
        """Detect Vietnamese administrative document structure"""
        structure = {
            'sections': [],  # Phần I, II, III
            'chapters': [],  # Chương 1, 2, 3  
            'articles': [],  # Điều 1, 2, 3
            'points': []     # 1., 2., a), b)
        }
        
        # Section pattern: Phần I, II, III or PHẦN I, II, III
        section_pattern = r'(?:PHẦN|Phần)\s+([IVX]+|[0-9]+)'
        structure['sections'] = re.findall(section_pattern, text, re.IGNORECASE)
        
        # Chapter pattern: Chương 1, 2, 3
        chapter_pattern = r'(?:CHƯƠNG|Chương)\s+([0-9]+)'
        structure['chapters'] = re.findall(chapter_pattern, text, re.IGNORECASE)
        
        # Article pattern: Điều 1, 2, 3
        article_pattern = r'(?:ĐIỀU|Điều)\s+([0-9]+)'
        structure['articles'] = re.findall(article_pattern, text, re.IGNORECASE)
        
        # Point pattern: 1., 2., a), b), c)
        point_pattern = r'^([0-9]+\.|[a-z]\))'
        for line in text.split('\n'):
            if re.match(point_pattern, line.strip()):
                structure['points'].append(line.strip())
        
        return structure

def generate_chunk_id(content: str, metadata: Dict = None) -> str:
    """
    Generate unique ID for chunk that's compatible with both FAISS and Qdrant.
    Returns a deterministic string ID that can be mapped to UUID for Qdrant.
    """
    # Create a deterministic hash based on content and key metadata
    content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    if metadata:
        # Use key metadata fields for uniqueness
        key_fields = [
            'document_id', 'document_title', 'section_title', 
            'start_line', 'end_line', 'chunk_level', 'chunk_type'
        ]
        
        meta_parts = []
        for field in key_fields:
            if field in metadata and metadata[field]:
                meta_parts.append(f"{field}:{str(metadata[field])}")
        
        if meta_parts:
            meta_str = "|".join(meta_parts)
            meta_hash = hashlib.sha256(meta_str.encode('utf-8')).hexdigest()[:8]
            return f"{content_hash}_{meta_hash}"
    
    return content_hash

def generate_deterministic_uuid(chunk_id: str) -> str:
    """
    Generate a deterministic UUID for a chunk ID.
    This ensures the same chunk_id always gets the same UUID.
    """
    # Use a fixed namespace UUID for consistency
    namespace = uuid.UUID('12345678-1234-5678-1234-123456789abc')
    return str(uuid.uuid5(namespace, chunk_id))

def estimate_tokens(text: str) -> int:
    """Estimate token count for Vietnamese text"""
    # Vietnamese specific estimation (roughly 1.2 tokens per word)
    words = len(text.split())
    return int(words * 1.2)

def validate_qdrant_point_id(point_id: str) -> bool:
    """Validate if a string is a valid Qdrant point ID (UUID or integer)"""
    # Check if it's a valid UUID
    try:
        uuid.UUID(point_id)
        return True
    except ValueError:
        pass
    
    # Check if it's a valid integer
    try:
        int(point_id)
        return True
    except ValueError:
        pass
    
    return False

def create_qdrant_compatible_id(original_id: str) -> str:
    """
    Create a Qdrant-compatible point ID from any string ID.
    Returns a UUID string that can be used as Qdrant point ID.
    """
    if validate_qdrant_point_id(original_id):
        return original_id
    
    # Generate deterministic UUID
    return generate_deterministic_uuid(original_id)

def safe_filename(text: str, max_length: int = 100) -> str:
    """Create a safe filename from text"""
    # Remove or replace unsafe characters
    safe_text = re.sub(r'[^\w\s-]', '', text)
    safe_text = re.sub(r'[-\s]+', '-', safe_text)
    
    # Truncate if too long
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length]
    
    return safe_text.strip('-')

def extract_document_metadata(file_path: str, content: str = None) -> Dict:
    """Extract metadata from document file path and content"""
    import os
    from pathlib import Path
    
    metadata = {
        'file_path': file_path,
        'file_name': os.path.basename(file_path),
        'file_extension': Path(file_path).suffix.lower(),
        'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
    }
    
    # Extract document type from filename patterns
    filename_lower = metadata['file_name'].lower()
    
    if any(keyword in filename_lower for keyword in ['quyết định', 'qd', 'decision']):
        metadata['document_type'] = 'quyết_định'
    elif any(keyword in filename_lower for keyword in ['thông tư', 'tt', 'circular']):
        metadata['document_type'] = 'thông_tư'
    elif any(keyword in filename_lower for keyword in ['nghị định', 'nd', 'decree']):
        metadata['document_type'] = 'nghị_định'
    elif any(keyword in filename_lower for keyword in ['luật', 'law']):
        metadata['document_type'] = 'luật'
    else:
        metadata['document_type'] = 'văn_bản'
    
    # Extract number from filename if present
    number_match = re.search(r'(\d+)', filename_lower)
    if number_match:
        metadata['document_number'] = number_match.group(1)
    
    # Extract year from filename if present
    year_match = re.search(r'(20\d{2})', filename_lower)
    if year_match:
        metadata['document_year'] = year_match.group(1)
    
    # Extract content-based metadata if content is provided
    if content:
        # Extract title from first few lines
        lines = content.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if len(line) > 10 and len(line) < 200:
                    metadata['extracted_title'] = line
                    break
        
        # Estimate content statistics
        metadata['content_length'] = len(content)
        metadata['estimated_tokens'] = estimate_tokens(content)
        metadata['line_count'] = len(content.split('\n'))
        metadata['paragraph_count'] = len([p for p in content.split('\n\n') if p.strip()])
    
    return metadata

def clean_vietnamese_text(text: str) -> str:
    """Clean and normalize Vietnamese text"""
    # Normalize diacritics
    text = VietnameseTextProcessor.normalize_diacritics(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\w\s\-.,;:!?(){}[\]""''`~@#$%^&*+=<>/\\|]', '', text)
    
    # Clean up punctuation spacing
    text = re.sub(r'\s*([.,;:!?])\s*', r'\1 ', text)
    
    return text.strip()

def format_administrative_reference(admin_info: Dict) -> str:
    """Format administrative information as a readable reference"""
    parts = []
    
    if admin_info.get('section'):
        parts.append(f"Phần {admin_info['section']}")
    if admin_info.get('chapter'):
        parts.append(f"Chương {admin_info['chapter']}")
    if admin_info.get('article'):
        parts.append(f"Điều {admin_info['article']}")
    if admin_info.get('point'):
        parts.append(f"Điểm {admin_info['point']}")
    if admin_info.get('subpoint'):
        parts.append(f"Tiểu điểm {admin_info['subpoint']}")
    
    return " > ".join(parts) if parts else ""

def create_search_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract key search terms from Vietnamese text"""
    # Normalize text
    text = clean_vietnamese_text(text.lower())
    
    # Remove common Vietnamese stop words
    stop_words = {
        'là', 'của', 'trong', 'với', 'và', 'các', 'một', 'có', 'được', 'này',
        'đó', 'cho', 'từ', 'theo', 'về', 'để', 'khi', 'nhưng', 'mà', 'hay',
        'hoặc', 'nếu', 'thì', 'sẽ', 'đã', 'đang', 'sẽ', 'phải', 'cần', 'nên',
        'những', 'nhiều', 'ít', 'lớn', 'nhỏ', 'cao', 'thấp', 'tốt', 'xấu'
    }
    
    # Extract words
    words = re.findall(r'\b\w{3,}\b', text)
    
    # Filter out stop words and count frequency
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) >= 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:max_keywords]]

def validate_chunk_metadata(metadata: Dict) -> bool:
    """Validate chunk metadata for completeness"""
    required_fields = ['document_id', 'chunk_type', 'content_type']
    
    for field in required_fields:
        if field not in metadata or not metadata[field]:
            logger.warning(f"Missing required metadata field: {field}")
            return False
    
    # Validate chunk_type
    valid_chunk_types = ['parent', 'child', 'table_parent', 'table_child']
    if metadata['chunk_type'] not in valid_chunk_types:
        logger.warning(f"Invalid chunk_type: {metadata['chunk_type']}")
        return False
    
    return True

def merge_chunk_metadata(base_metadata: Dict, additional_metadata: Dict) -> Dict:
    """Merge two metadata dictionaries with conflict resolution"""
    merged = base_metadata.copy()
    
    for key, value in additional_metadata.items():
        if key in merged:
            # Handle conflicts
            if key == 'keywords':
                # Merge keyword lists
                base_keywords = merged.get('keywords', [])
                additional_keywords = value if isinstance(value, list) else []
                merged['keywords'] = list(set(base_keywords + additional_keywords))
            elif key.endswith('_info') and isinstance(value, dict):
                # Merge info dictionaries
                if isinstance(merged[key], dict):
                    merged[key].update(value)
                else:
                    merged[key] = value
            else:
                # Prefer additional metadata for other fields
                merged[key] = value
        else:
            merged[key] = value
    
    return merged

def estimate_processing_time(content_length: int, chunk_count: int = None) -> float:
    """Estimate processing time for document content"""
    # Base time estimates (in seconds)
    base_time_per_char = 0.00001  # Very rough estimate
    base_time_per_chunk = 0.1
    embedding_time_per_chunk = 0.05
    
    time_estimate = content_length * base_time_per_char
    
    if chunk_count:
        time_estimate += chunk_count * (base_time_per_chunk + embedding_time_per_chunk)
    else:
        # Estimate chunk count
        estimated_chunks = max(1, content_length // 1000)
        time_estimate += estimated_chunks * (base_time_per_chunk + embedding_time_per_chunk)
    
    return time_estimate

def create_progress_callback(total_items: int, description: str = "Processing"):
    """Create a progress callback function for long-running operations"""
    def progress_callback(current_item: int, item_description: str = ""):
        percentage = (current_item / total_items) * 100
        bar_length = 30
        filled_length = int(bar_length * current_item // total_items)
        
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r{description}: |{bar}| {percentage:.1f}% ({current_item}/{total_items}) {item_description}', end='')
        
        if current_item == total_items:
            print()  # New line when complete
    
    return progress_callback

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def debug_chunk_info(chunk, include_content: bool = False) -> str:
    """Create debug information string for a chunk"""
    info_parts = [
        f"ID: {chunk.id}",
        f"Type: {chunk.chunk_type}",
        f"Tokens: {chunk.tokens}",
        f"Parent: {chunk.parent_id if chunk.parent_id else 'None'}",
        f"Children: {len(chunk.child_ids) if chunk.child_ids else 0}"
    ]
    
    if chunk.metadata:
        doc_title = chunk.metadata.get('document_title', 'Unknown')
        section_title = chunk.metadata.get('section_title', 'Unknown')
        info_parts.extend([
            f"Document: {doc_title}",
            f"Section: {section_title}"
        ])
    
    debug_info = " | ".join(info_parts)
    
    if include_content:
        content_preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
        debug_info += f"\nContent: {content_preview}"
    
    return debug_info

def validate_embedding_dimension(embedding: np.ndarray, expected_dim: int) -> bool:
    """Validate embedding dimension"""
    import numpy as np
    
    if not isinstance(embedding, np.ndarray):
        logger.error(f"Embedding is not numpy array: {type(embedding)}")
        return False
    
    if len(embedding.shape) != 1:
        logger.error(f"Embedding should be 1D array, got shape: {embedding.shape}")
        return False
    
    if embedding.shape[0] != expected_dim:
        logger.error(f"Embedding dimension mismatch: expected {expected_dim}, got {embedding.shape[0]}")
        return False
    
    return True

def log_system_info():
    """Log system information for debugging"""
    import platform
    import psutil
    import sys
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    except ImportError:
        logger.info("PyTorch not installed")
    
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, timeout=5)
        collections = client.get_collections()
        logger.info(f"Qdrant: Connected, {len(collections.collections)} collections")
        client.close()
    except Exception as e:
        logger.warning(f"Qdrant connection failed: {e}")

# Example usage and testing functions
def test_chunk_id_generation():
    """Test chunk ID generation for Qdrant compatibility"""
    test_content = "This is a test content for chunk ID generation"
    test_metadata = {
        'document_id': 'test_doc_1',
        'section_title': 'Test Section',
        'chunk_type': 'parent'
    }
    
    # Generate chunk ID
    chunk_id = generate_chunk_id(test_content, test_metadata)
    print(f"Generated chunk ID: {chunk_id}")
    
    # Generate UUID for Qdrant
    uuid_id = create_qdrant_compatible_id(chunk_id)
    print(f"Qdrant UUID: {uuid_id}")
    
    # Validate
    is_valid = validate_qdrant_point_id(uuid_id)
    print(f"Valid Qdrant ID: {is_valid}")
    
    # Test deterministic generation
    chunk_id_2 = generate_chunk_id(test_content, test_metadata)
    uuid_id_2 = create_qdrant_compatible_id(chunk_id_2)
    
    print(f"Deterministic test: {chunk_id == chunk_id_2 and uuid_id == uuid_id_2}")

if __name__ == "__main__":
    # Run tests
    print("Testing chunk ID generation for Qdrant...")
    test_chunk_id_generation()
    
    print("\nTesting Vietnamese text processing...")
    sample_text = "Đây là một văn bản tiếng Việt để thử nghiệm."
    normalized = VietnameseTextProcessor.normalize_diacritics(sample_text)
    print(f"Original: {sample_text}")
    print(f"Normalized: {normalized}")
    
    print("\nTesting administrative structure detection...")
    admin_text = """
    PHẦN I: QUY ĐỊNH CHUNG
    CHƯƠNG 1: Phạm vi áp dụng
    Điều 1. Đối tượng áp dụng
    1. Cá nhân cư trú
    a) Người có quốc tịch Việt Nam
    """
    structure = VietnameseTextProcessor.detect_administrative_structure(admin_text)
    print(f"Detected structure: {structure}")