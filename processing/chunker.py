from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import csv
from typing import List, Dict, Optional, Union
import logging
import pandas as pd

from core.config import ChunkingConfig
from utils.utils import estimate_tokens, generate_chunk_id, VietnameseTextProcessor
from processing.preprocessor import DocumentStructure, HeaderInfo


logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class Chunk:
    id: str
    content: str
    chunk_type: str  # 'parent', 'child', 'table_parent', 'table_child'
    metadata: Dict
    parent_id: Optional[str] = None
    child_ids: List[str] = None
    tokens: int = 0
    
    def __post_init__(self):
        if self.child_ids is None:
            self.child_ids = []
        if self.tokens == 0:
            self.tokens = estimate_tokens(self.content)

class HierarchicalChunker:
    """Hierarchical chunking for Vietnamese documents with enhanced header metadata"""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.text_processor = VietnameseTextProcessor()
    
    def chunk_document(self, doc_structure: DocumentStructure, doc_metadata: Dict = None) -> List[Chunk]:
        """Main chunking pipeline with enhanced header metadata and document metadata from Excel"""
        all_chunks = []
        
        # Use document metadata from Excel if provided
        document_metadata = doc_metadata or {}
        
        for section in doc_structure.sections:
            # Find header context for this section
            header_context = self._get_section_header_context(section, doc_structure.header_hierarchy)
            
            # Determine if section contains tables
            section_tables = self._get_section_tables(section, doc_structure.tables)
            
            if section_tables:
                chunks = self._chunk_section_with_tables(section, section_tables, doc_structure, header_context, document_metadata)
            else:
                chunks = self._chunk_text_section(section, doc_structure, header_context, document_metadata)
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _get_section_header_context(self, section: Dict, header_hierarchy: List[HeaderInfo]) -> Optional[HeaderInfo]:
        """Get the header context for a section"""
        section_start_line = section.get('start_line', 0)
        
        # Find the header that corresponds to this section
        for header in header_hierarchy:
            if header.line_number == section_start_line or (
                header.line_number <= section_start_line and 
                header.title == section.get('title')
            ):
                return header
        
        return None
    
    def _chunk_section_with_tables(self, section: Dict, tables: List[Dict], 
                                  doc_structure: DocumentStructure, header_context: Optional[HeaderInfo],
                                  document_metadata: Dict) -> List[Chunk]:
        """Chunk section containing tables with enhanced metadata"""
        chunks = []
        
        # Create table-aware parent chunk
        parent_content = self._create_table_aware_parent_content(section, tables)
        parent_metadata = self._create_enhanced_parent_metadata(section, doc_structure, header_context, has_tables=True, document_metadata=document_metadata)
        
        parent_chunk = Chunk(
            id=generate_chunk_id(parent_content, parent_metadata),
            content=parent_content,
            chunk_type='table_parent',
            metadata=parent_metadata
        )
        
        # Create child chunks
        child_chunks = self._create_table_aware_children(section, tables, parent_chunk.id, doc_structure, header_context, document_metadata)
        
        # Update parent with child IDs
        parent_chunk.child_ids = [child.id for child in child_chunks]
        
        chunks.append(parent_chunk)
        chunks.extend(child_chunks)
        
        return chunks
    
    def _chunk_text_section(self, section: Dict, doc_structure: DocumentStructure, 
                           header_context: Optional[HeaderInfo], document_metadata: Dict) -> List[Chunk]:
        """Chunk text-only section with enhanced metadata"""
        chunks = []
        
        # Create parent chunk
        parent_content = self._create_contextual_content(section, doc_structure, header_context, document_metadata)
        parent_metadata = self._create_enhanced_parent_metadata(section, doc_structure, header_context, has_tables=False, document_metadata=document_metadata)
        
        parent_chunk = Chunk(
            id=generate_chunk_id(parent_content, parent_metadata),
            content=parent_content,
            chunk_type='parent',
            metadata=parent_metadata
        )
        
        # Create child chunks
        child_chunks = self._create_text_children(section, parent_chunk.id, doc_structure, header_context, document_metadata)
        
        # Update parent with child IDs
        parent_chunk.child_ids = [child.id for child in child_chunks]
        
        chunks.append(parent_chunk)
        chunks.extend(child_chunks)
        
        return chunks
    
    def _create_enhanced_parent_metadata(self, section: Dict, doc_structure: DocumentStructure, 
                                    header_context: Optional[HeaderInfo], has_tables: bool,
                                    document_metadata: Dict) -> Dict:
        """Create enhanced metadata for parent chunk with sanitized values"""
        metadata = {
            'chunk_level': 'parent',
            'section_title': section.get('title', ''),
            'section_level': section.get('level', 0),
            'has_tables': has_tables,
            'administrative_info': section.get('administrative_info', {}),
            'start_line': section.get('start_line', 0),
            'end_line': section.get('end_line', 0),
            'section_type': section.get('type', 'unknown')
        }
        
        # Add document metadata from Excel (sanitized)
        if document_metadata:
            sanitized_doc_metadata = self._sanitize_document_metadata(document_metadata)
            metadata.update(sanitized_doc_metadata)
        else:
            # Fallback to doc_structure if no document_metadata
            metadata['document_title'] = doc_structure.title
        
        # Add enhanced header metadata
        if header_context:
            metadata['header_metadata'] = {
                'full_path': header_context.full_path,
                'parent_headers': header_context.parent_headers,
                'header_level': header_context.level,
                'raw_header_text': header_context.raw_text,
                'header_line_number': header_context.line_number,
                'header_administrative_info': header_context.administrative_info
            }
        
        # Add section header metadata if available
        section_header_metadata = section.get('header_metadata', {})
        if section_header_metadata:
            metadata['section_header_metadata'] = section_header_metadata
        
        # Build hierarchy path
        hierarchy_path = self._build_enhanced_hierarchy_path(section, header_context)
        metadata['hierarchy_path'] = hierarchy_path
        
        # Add breadcrumb navigation
        metadata['breadcrumb'] = self._create_breadcrumb(header_context)
        
        return metadata
    
    def _create_enhanced_child_metadata(self, section: Dict, doc_structure: DocumentStructure,
                                    chunk_index: int, content_type: str, 
                                    header_context: Optional[HeaderInfo],
                                    document_metadata: Dict) -> Dict:
        """Create enhanced metadata for child chunk with sanitized values"""
        metadata = {
            'chunk_level': 'child',
            'chunk_index': chunk_index,
            'content_type': content_type,
            'section_title': section.get('title', ''),
            'section_level': section.get('level', 0),
            'administrative_info': section.get('administrative_info', {}),
            'section_type': section.get('type', 'unknown')
        }
        
        # Add document metadata from Excel (sanitized)
        if document_metadata:
            sanitized_doc_metadata = self._sanitize_document_metadata(document_metadata)
            metadata.update(sanitized_doc_metadata)
        else:
            # Fallback to doc_structure if no document_metadata
            metadata['document_title'] = doc_structure.title
        
        # Add header context if available
        if header_context:
            metadata['header_metadata'] = {
                'full_path': header_context.full_path,
                'parent_headers': header_context.parent_headers,
                'header_level': header_context.level,
                'header_administrative_info': header_context.administrative_info
            }
            
            # Add breadcrumb for child
            metadata['breadcrumb'] = self._create_breadcrumb(header_context)
        
        # Build hierarchy path for child
        hierarchy_path = self._build_enhanced_hierarchy_path(section, header_context)
        if chunk_index > 0:
            hierarchy_path += f" > Đoạn {chunk_index + 1}"
        metadata['hierarchy_path'] = hierarchy_path
        
        return metadata
    def _sanitize_document_metadata(self, document_metadata: Dict) -> Dict:
        """Sanitize document metadata to ensure JSON serialization"""
        sanitized = {}
        
        for key, value in document_metadata.items():
            if key == 'doc_date':
                # Special handling for doc_date
                sanitized[key] = self._sanitize_date_value(value)
            elif isinstance(value, (int, float, str, bool, type(None))):
                # Basic types are safe
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples, sanitizing each element
                sanitized[key] = [self._sanitize_value(item) for item in value]
            elif isinstance(value, dict):
                # Recursively sanitize dictionaries
                sanitized[key] = self._sanitize_document_metadata(value)
            else:
                # Convert complex objects to string
                sanitized[key] = str(value)
        
        return sanitized

    def _sanitize_date_value(self, date_value) -> str:
        """Sanitize date value specifically"""
        if not date_value or pd.isna(date_value):
            return 'Unknown'
        
        # If already a string, return as is
        if isinstance(date_value, str):
            return date_value.strip()
        
        # Handle pandas Timestamp
        try:
            if isinstance(date_value, pd.Timestamp):
                return date_value.strftime('%Y-%m-%d')
        except Exception:
            pass
        
        # Handle datetime objects
        try:
            import datetime
            if isinstance(date_value, (datetime.datetime, datetime.date)):
                return date_value.strftime('%Y-%m-%d')
        except Exception:
            pass
        
        # Handle numpy datetime
        try:
            import numpy as np
            if isinstance(date_value, np.datetime64):
                return str(date_value)[:10]
        except Exception:
            pass
        
        # Fallback
        try:
            return str(date_value).strip()
        except Exception:
            return 'Unknown'

    def _sanitize_value(self, value):
        """Sanitize individual value for JSON serialization"""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif hasattr(value, '__dict__'):
            # Object with attributes - convert to dict
            return {k: self._sanitize_value(v) for k, v in value.__dict__.items()}
        else:
            return str(value)
        
    def _build_enhanced_hierarchy_path(self, section: Dict, header_context: Optional[HeaderInfo]) -> str:
        """Build enhanced hierarchy path with full header context"""
        if header_context and header_context.full_path:
            return header_context.full_path
        
        # Fallback to administrative info
        admin_info = section.get('administrative_info', {})
        path_parts = []
        
        if 'section' in admin_info:
            path_parts.append(f"Phần {admin_info['section']}")
        if 'chapter' in admin_info:
            path_parts.append(f"Chương {admin_info['chapter']}")
        if 'article' in admin_info:
            path_parts.append(f"Điều {admin_info['article']}")
        if 'point' in admin_info:
            path_parts.append(f"Điểm {admin_info['point']}")
        if 'subpoint' in admin_info:
            path_parts.append(f"Điểm {admin_info['subpoint']}")
        
        if not path_parts and section.get('title'):
            path_parts.append(section['title'])
        
        return ' > '.join(path_parts) if path_parts else 'Không xác định'
    
    def _create_breadcrumb(self, header_context: Optional[HeaderInfo]) -> List[Dict]:
        """Create breadcrumb navigation from header context"""
        if not header_context:
            return []
        
        breadcrumb = []
        
        # Add parent headers
        for parent in header_context.parent_headers:
            breadcrumb.append({
                'title': parent['title'],
                'level': parent['level'],
                'line_number': parent['line_number'],
                'administrative_info': parent.get('administrative_info', {})
            })
        
        # Add current header
        breadcrumb.append({
            'title': header_context.title,
            'level': header_context.level,
            'line_number': header_context.line_number,
            'administrative_info': header_context.administrative_info
        })
        
        return breadcrumb
    
    def _create_table_aware_parent_content(self, section: Dict, tables: List[Dict]) -> str:
        """Create parent content that includes table context with enhanced metadata"""
        content_parts = []
        
        # Add section title and context
        if section.get('title'):
            content_parts.append(f"# {section['title']}")
        
        section_content = section.get('content', '')
        
        # Process content with table insertions
        lines = section_content.split('\n')
        current_content = []
        
        for i, line in enumerate(lines):
            # Check if this line starts a table
            table_at_line = self._find_table_at_line(i + section.get('start_line', 0), tables)
            
            if table_at_line:
                # Add accumulated content
                if current_content:
                    content_parts.append('\n'.join(current_content))
                    current_content = []
                
                # Add table with enhanced context
                table_context = self._create_enhanced_table_context(table_at_line, section)
                content_parts.append(table_context)
                
                # Skip table lines
                table_line_count = table_at_line['end_line'] - table_at_line['start_line'] + 1
                for _ in range(table_line_count):
                    if i < len(lines):
                        i += 1
            else:
                current_content.append(line)
        
        # Add remaining content
        if current_content:
            content_parts.append('\n'.join(current_content))
        
        return '\n\n'.join(content_parts)
    
    def _create_enhanced_table_context(self, table: Dict, section: Dict) -> str:
        """Create enhanced context for table with header information"""
        context_parts = []
        
        # Add table summary with header context
        parsed_data = table.get('parsed_data', {})
        headers = parsed_data.get('headers', [])
        row_count = table.get('row_count', 0)
        
        summary = f"**Bảng ({row_count} hàng, {len(headers)} cột)**"
        if headers:
            summary += f" - Các cột: {', '.join(headers)}"
        
        # Add header context information
        header_context = table.get('header_context', {})
        if header_context and header_context.get('title'):
            summary += f" - Thuộc mục: {header_context['title']}"
        
        context_parts.append(summary)
        context_parts.append(table.get('raw_content', ''))
        
        return '\n'.join(context_parts)
    
    def _create_table_aware_children(self, section: Dict, tables: List[Dict], 
                                   parent_id: str, doc_structure: DocumentStructure,
                                   header_context: Optional[HeaderInfo],
                                   document_metadata: Dict) -> List[Chunk]:
        """Create child chunks for section with tables"""
        children = []
        
        # Text-only children
        text_content = self._extract_text_without_tables(section, tables)
        if text_content.strip():
            text_children = self._split_text_into_chunks(
                text_content, 
                self.config.child_chunk_size,
                self.config.overlap_tokens
            )
            
            for i, chunk_text in enumerate(text_children):
                child_metadata = self._create_enhanced_child_metadata(
                    section, doc_structure, i, 'text', header_context, document_metadata
                )
                child = Chunk(
                    id=generate_chunk_id(chunk_text, child_metadata),
                    content=chunk_text,
                    chunk_type='child',
                    metadata=child_metadata,
                    parent_id=parent_id
                )
                children.append(child)
        
        # Table-specific children
        for table in tables:
            table_children = self._create_enhanced_table_children(
                table, parent_id, section, doc_structure, header_context, document_metadata
            )
            children.extend(table_children)
        
        return children
    
    def _create_text_children(self, section: Dict, parent_id: str, 
                            doc_structure: DocumentStructure,
                            header_context: Optional[HeaderInfo],
                            document_metadata: Dict) -> List[Chunk]:
        """Create child chunks for text-only section with enhanced metadata"""
        children = []
        
        content = section.get('content', '')
        if not content.strip():
            return children
        
        # Split into child-sized chunks
        text_chunks = self._split_text_into_chunks(
            content,
            self.config.child_chunk_size, 
            self.config.overlap_tokens
        )
        
        for i, chunk_text in enumerate(text_chunks):
            # Add contextual information
            contextual_content = self._add_enhanced_child_context(
                chunk_text, section, doc_structure, header_context, document_metadata
            )
            
            child_metadata = self._create_enhanced_child_metadata(
                section, doc_structure, i, 'text', header_context, document_metadata
            )
            child = Chunk(
                id=generate_chunk_id(contextual_content, child_metadata),
                content=contextual_content,
                chunk_type='child',
                metadata=child_metadata,
                parent_id=parent_id
            )
            children.append(child)
        
        return children
    
    def _add_enhanced_child_context(self, content: str, section: Dict, 
                                   doc_structure: DocumentStructure,
                                   header_context: Optional[HeaderInfo],
                                   document_metadata: Dict) -> str:
        """Add enhanced contextual information to child chunk"""
        context_parts = []
        
        # Document context - prioritize metadata from Excel
        if document_metadata and document_metadata.get('doc_title'):
            context_parts.append(f"Tài liệu: {document_metadata['doc_title']}")
        else:
            context_parts.append(f"Tài liệu: {doc_structure.title}")
        
        # Enhanced hierarchy context
        if header_context and header_context.full_path:
            context_parts.append(f"Vị trí: {header_context.full_path}")
        elif section.get('title'):
            context_parts.append(f"Phần: {section['title']}")
        
        # Administrative context with more detail
        admin_info = section.get('administrative_info', {})
        if admin_info:
            admin_context = []
            if 'section_full' in admin_info:
                admin_context.append(admin_info['section_full'])
            elif 'section' in admin_info:
                admin_context.append(f"Phần {admin_info['section']}")
                
            if 'chapter_full' in admin_info:
                admin_context.append(admin_info['chapter_full'])
            elif 'chapter' in admin_info:
                admin_context.append(f"Chương {admin_info['chapter']}")
                
            if 'article_full' in admin_info:
                admin_context.append(admin_info['article_full'])
            elif 'article' in admin_info:
                admin_context.append(f"Điều {admin_info['article']}")
            
            if admin_context:
                context_parts.append(" > ".join(admin_context))
        
        # Combine with content
        context_prefix = "**Bối cảnh:** " + " | ".join(context_parts) + "\n\n"
        return context_prefix + content
    
    def _create_enhanced_table_children(self, table: Dict, parent_id: str, section: Dict,
                                       doc_structure: DocumentStructure,
                                       header_context: Optional[HeaderInfo],
                                       document_metadata: Dict) -> List[Chunk]:
        """Create child chunks for table with enhanced metadata"""
        children = []
        
        parsed_data = table.get('parsed_data', {})
        headers = parsed_data.get('headers', [])
        rows = parsed_data.get('rows', [])
        
        # Table summary child with enhanced context
        summary_content = self._create_enhanced_table_summary(table, section, header_context, document_metadata)
        summary_metadata = self._create_enhanced_child_metadata(
            section, doc_structure, 0, 'table_summary', header_context, document_metadata
        )
        
        # Add table-specific metadata
        summary_metadata['table_info'] = {
            'headers': headers,
            'row_count': len(rows),
            'column_count': len(headers),
            'table_header_context': table.get('header_context', {}),
            'table_line_number': table.get('start_line', 0)
        }
        
        summary_child = Chunk(
            id=generate_chunk_id(summary_content, summary_metadata),
            content=summary_content,
            chunk_type='table_child',
            metadata=summary_metadata,
            parent_id=parent_id
        )
        children.append(summary_child)
        
        # Table data child(s) with enhanced metadata
        if len(rows) <= 10:  # Small table - single chunk
            data_content = self._format_enhanced_table_data(headers, rows, table, header_context, document_metadata)
            data_metadata = self._create_enhanced_child_metadata(
                section, doc_structure, 1, 'table_data', header_context, document_metadata
            )
            data_metadata['table_info'] = summary_metadata['table_info'].copy()
            
            data_child = Chunk(
                id=generate_chunk_id(data_content, data_metadata),
                content=data_content,
                chunk_type='table_child',
                metadata=data_metadata,
                parent_id=parent_id
            )
            children.append(data_child)
        
        else:  # Large table - split into multiple chunks
            chunk_size = 5  # rows per chunk
            for i in range(0, len(rows), chunk_size):
                chunk_rows = rows[i:i + chunk_size]
                data_content = self._format_enhanced_table_data(headers, chunk_rows, table, header_context, document_metadata)
                data_metadata = self._create_enhanced_child_metadata(
                    section, doc_structure, i + 1, 'table_data', header_context, document_metadata
                )
                data_metadata['table_info'] = {
                    'headers': headers,
                    'row_count': len(chunk_rows),
                    'column_count': len(headers),
                    'chunk_start_row': i,
                    'chunk_end_row': i + len(chunk_rows) - 1,
                    'table_header_context': table.get('header_context', {}),
                    'table_line_number': table.get('start_line', 0)
                }
                
                data_child = Chunk(
                    id=generate_chunk_id(data_content, data_metadata),
                    content=data_content,
                    chunk_type='table_child',
                    metadata=data_metadata,
                    parent_id=parent_id
                )
                children.append(data_child)
        
        return children
    
    def _create_enhanced_table_summary(self, table: Dict, section: Dict, 
                                      header_context: Optional[HeaderInfo],
                                      document_metadata: Dict) -> str:
        """Create enhanced summary description of table with header context"""
        parsed_data = table.get('parsed_data', {})
        headers = parsed_data.get('headers', [])
        rows = parsed_data.get('rows', [])
        
        summary_parts = []
        
        # Enhanced location context
        if header_context and header_context.full_path:
            summary_parts.append(f"Bảng trong {header_context.full_path}")
        else:
            summary_parts.append(f"Bảng trong {section.get('title', 'phần này')}")
        
        # Basic info
        summary_parts.append(f"có {len(rows)} hàng và {len(headers)} cột.")
        
        # Header context from table
        table_header_context = table.get('header_context', {})
        if table_header_context and table_header_context.get('title'):
            summary_parts.append(f"Thuộc mục: {table_header_context['title']}")
        
        # Column info
        if headers:
            summary_parts.append(f"Các cột bao gồm: {', '.join(headers)}")
        
        # Sample data
        if rows:
            summary_parts.append("Dữ liệu mẫu:")
            for i, row in enumerate(rows[:3]):  # First 3 rows
                row_data = []
                for j, cell in enumerate(row):
                    if j < len(headers):
                        row_data.append(f"{headers[j]}: {cell}")
                summary_parts.append(f"- Hàng {i+1}: {'; '.join(row_data)}")
            
            if len(rows) > 3:
                summary_parts.append(f"... và {len(rows) - 3} hàng khác")
        
        return '\n'.join(summary_parts)
    
    def _format_enhanced_table_data(self, headers: List[str], rows: List[List[str]], 
                                   table: Dict, header_context: Optional[HeaderInfo],
                                   document_metadata: Dict) -> str:
        """Format table data as text with enhanced context"""
        if not headers or not rows:
            return ""
        
        lines = []
        
        # Add context header
        if header_context and header_context.full_path:
            lines.append(f"**Bảng từ: {header_context.full_path}**\n")
        
        # Create formatted table
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join([" --- " for _ in headers]) + "|")
        
        # Rows
        for row in rows:
            # Pad row to match header count
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")
        
        return '\n'.join(lines)
    
    def _create_contextual_content(self, section: Dict, doc_structure: DocumentStructure,
                                  header_context: Optional[HeaderInfo],
                                  document_metadata: Dict) -> str:
        """Create enhanced contextual content for parent chunk"""
        content_parts = []
        
        # Add enhanced hierarchical context
        if header_context and header_context.full_path:
            content_parts.append(f"**Vị trí trong tài liệu:** {header_context.full_path}")
        else:
            hierarchy_path = self._build_enhanced_hierarchy_path(section, header_context)
            if hierarchy_path and hierarchy_path != 'Không xác định':
                content_parts.append(f"**Vị trí trong tài liệu:** {hierarchy_path}")
        
        # Add section title
        if section.get('title'):
            content_parts.append(f"# {section['title']}")
        
        # Add administrative context if available
        admin_info = section.get('administrative_info', {})
        if admin_info:
            admin_details = []
            for key, value in admin_info.items():
                if key.endswith('_full') and value:
                    admin_details.append(value)
            if admin_details:
                content_parts.append(f"**Cấu trúc hành chính:** {' > '.join(admin_details)}")
        
        # Add main content
        content_parts.append(section.get('content', ''))
        
        return '\n\n'.join(content_parts)
    
    # Existing utility methods remain the same...
    def _split_text_into_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if not text.strip():
            return []
        
        # Estimate current tokens
        current_tokens = estimate_tokens(text)
        
        if current_tokens <= chunk_size:
            return [text]
        
        # Split by sentences to respect boundaries
        sentences = self.text_processor.extract_vietnamese_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk, overlap)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(estimate_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get sentences for overlap based on token count"""
        if not sentences:
            return []
        
        overlap_sentences = []
        total_tokens = 0
        
        # Work backwards from end of chunk
        for sentence in reversed(sentences):
            sentence_tokens = estimate_tokens(sentence)
            if total_tokens + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                total_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _get_section_tables(self, section: Dict, all_tables: List[Dict]) -> List[Dict]:
        """Get tables that belong to this section"""
        section_start = section.get('start_line', 0)
        section_end = section.get('end_line', float('inf'))
        
        section_tables = []
        for table in all_tables:
            table_start = table.get('start_line', 0)
            if section_start <= table_start <= section_end:
                section_tables.append(table)
        
        return section_tables
    
    def _find_table_at_line(self, line_num: int, tables: List[Dict]) -> Optional[Dict]:
        """Find table that starts at given line"""
        for table in tables:
            if table.get('start_line', 0) == line_num:
                return table
        return None
    
    def _extract_text_without_tables(self, section: Dict, tables: List[Dict]) -> str:
        """Extract text content excluding tables"""
        content = section.get('content', '')
        lines = content.split('\n')
        
        # Remove table lines
        filtered_lines = []
        section_start = section.get('start_line', 0)
        
        for i, line in enumerate(lines):
            line_num = section_start + i
            
            # Check if this line is part of a table
            is_table_line = False
            for table in tables:
                if table.get('start_line', 0) <= line_num <= table.get('end_line', 0):
                    is_table_line = True
                    break
            
            if not is_table_line:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    # Save chunk to file csv to check parent-child relationships
    def save_chunks_to_csv(self, chunks: List[Chunk], file_path: str):
        """ Lưu danh sách các chunk vào file CSV để kiêm tra quan hệ parent-child """
       
        with open(file_path, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Content', 'Chunk Type', 'Parent ID', 'Child IDs', 'Tokens', 'Metadata'])
            
            for chunk in chunks:
                child_ids_str = ', '.join(chunk.child_ids) if chunk.child_ids else ''
                writer.writerow([
                    chunk.id,
                    chunk.content,
                    chunk.chunk_type,
                    chunk.parent_id,
                    child_ids_str,
                    chunk.tokens,
                    json.dumps(chunk.metadata, ensure_ascii=False)
                ])
        
        logger.info(f"Chunks saved to {file_path}")