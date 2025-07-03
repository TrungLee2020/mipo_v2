import markdown
# from markdown.extensions.tables import TableExtension
# from markdown.extensions.toc import TocExtension
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from utils.utils import VietnameseTextProcessor
from metadata import MetadataManager

@dataclass
class HeaderInfo:
    """Detailed header information"""
    level: int
    title: str
    full_path: str  # Hierarchy path from root
    line_number: int
    raw_text: str
    administrative_info: Dict
    parent_headers: List[Dict]  # Chain of parent headers

@dataclass
class DocumentStructure:
    title: str
    sections: List[Dict]
    tables: List[Dict]
    references: List[str]
    metadata: Dict
    header_hierarchy: List[HeaderInfo]  # Complete header structure

class VietnameseMarkdownPreprocessor:
    """Enhanced preprocess Vietnamese markdown documents with detailed header metadata"""
    
    def __init__(self):
        self.text_processor = VietnameseTextProcessor()
        self.md_parser = markdown.Markdown(
            extensions=['tables', 'toc', 'codehilite'],
            extension_configs={
                'toc': {'title': 'Mục lục'}
            }
        )
    
    def preprocess_document(self, content: str) -> DocumentStructure:
        """Main preprocessing pipeline with enhanced header tracking"""
        # Normalize text
        normalized_content = self.text_processor.normalize_diacritics(content)
        
        # Extract complete header hierarchy first
        header_hierarchy = self._extract_header_hierarchy(normalized_content)
        
        # Extract document structure with header metadata
        structure = self._extract_document_structure_with_headers(normalized_content, header_hierarchy)
        
        # Extract tables
        tables = self._extract_tables(normalized_content)
        
        # Extract references
        references = self._extract_references(normalized_content)
        
        # Extract metadata
        metadata = self._extract_metadata(normalized_content)
        
        return DocumentStructure(
            title=metadata.get('title', 'Untitled'),
            sections=structure,
            tables=tables,
            references=references,
            metadata=metadata,
            header_hierarchy=header_hierarchy
        )
    
    def _extract_header_hierarchy(self, content: str) -> List[HeaderInfo]:
        """Extract complete header hierarchy with full metadata"""
        headers = []
        lines = content.split('\n')
        header_stack = []  # Stack to track hierarchy
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Detect markdown headers
            if line_stripped.startswith('#'):
                header_info = self._parse_markdown_header(line_stripped, i, header_stack)
                headers.append(header_info)
                self._update_header_stack(header_stack, header_info)
                
            # Detect administrative headers
            elif self._is_administrative_header(line_stripped):
                header_info = self._parse_administrative_header(line_stripped, i, header_stack)
                headers.append(header_info)
                self._update_header_stack(header_stack, header_info)
        
        return headers
    
    def _parse_markdown_header(self, line: str, line_number: int, header_stack: List[HeaderInfo]) -> HeaderInfo:
        """Parse markdown header with full metadata"""
        level = len(line) - len(line.lstrip('#'))
        title = line.lstrip('#').strip()
        
        # Build full path
        full_path = self._build_header_path(header_stack, title, level)
        
        # Extract administrative info
        admin_info = self._extract_administrative_info(title)
        
        # Get parent headers
        parent_headers = self._get_parent_headers(header_stack, level)
        
        return HeaderInfo(
            level=level,
            title=title,
            full_path=full_path,
            line_number=line_number,
            raw_text=line,
            administrative_info=admin_info,
            parent_headers=parent_headers
        )
    
    def _parse_administrative_header(self, line: str, line_number: int, header_stack: List[HeaderInfo]) -> HeaderInfo:
        """Parse Vietnamese administrative header"""
        # Determine administrative level
        admin_level = self._determine_admin_level(line)
        
        # Build full path
        full_path = self._build_header_path(header_stack, line, admin_level)
        
        # Extract administrative info
        admin_info = self._extract_administrative_info(line)
        
        # Get parent headers
        parent_headers = self._get_parent_headers(header_stack, admin_level)
        
        return HeaderInfo(
            level=admin_level,
            title=line,
            full_path=full_path,
            line_number=line_number,
            raw_text=line,
            administrative_info=admin_info,
            parent_headers=parent_headers
        )
    
    def _build_header_path(self, header_stack: List[HeaderInfo], current_title: str, current_level: int) -> str:
        """Build hierarchical path for header"""
        path_parts = []
        
        # Add relevant parent headers
        for header in header_stack:
            if header.level < current_level:
                path_parts.append(header.title)
        
        # Add current header
        path_parts.append(current_title)
        
        return " > ".join(path_parts)
    
    def _get_parent_headers(self, header_stack: List[HeaderInfo], current_level: int) -> List[Dict]:
        """Get parent headers for current header"""
        parents = []
        for header in header_stack:
            if header.level < current_level:
                parents.append({
                    'level': header.level,
                    'title': header.title,
                    'line_number': header.line_number,
                    'administrative_info': header.administrative_info
                })
        return parents
    
    def _update_header_stack(self, header_stack: List[HeaderInfo], new_header: HeaderInfo):
        """Update header stack maintaining hierarchy"""
        # Remove headers at same or deeper level
        header_stack[:] = [h for h in header_stack if h.level < new_header.level]
        # Add new header
        header_stack.append(new_header)
    
    def _is_administrative_header(self, line: str) -> bool:
        """Check if line is administrative header"""
        admin_patterns = [
            r'^\s*(?:PHẦN|Phần)\s+[IVX]+\.',
            r'^\s*(?:CHƯƠNG|Chương)\s+[0-9]+\.',
            r'^\s*(?:ĐIỀU|Điều)\s+[0-9]+\.',
            r'^\s*[0-9]+\.\s+',
            r'^\s*[a-z]\)\s+'
        ]
        
        return any(re.match(pattern, line, re.IGNORECASE) for pattern in admin_patterns)
    
    def _extract_document_structure_with_headers(self, content: str, header_hierarchy: List[HeaderInfo]) -> List[Dict]:
        """Extract document structure with enhanced header metadata"""
        sections = []
        lines = content.split('\n')
        
        # Create sections based on headers
        for i, header in enumerate(header_hierarchy):
            start_line = header.line_number
            
            # Find end line (next header or end of document)
            end_line = len(lines) - 1
            if i + 1 < len(header_hierarchy):
                end_line = header_hierarchy[i + 1].line_number - 1
            
            # Extract content between headers
            section_lines = lines[start_line + 1:end_line + 1]
            content_text = '\n'.join(section_lines)
            
            # Create enhanced section metadata
            section = {
                'level': header.level,
                'title': header.title,
                'start_line': start_line,
                'end_line': end_line,
                'content': content_text,
                'administrative_info': header.administrative_info,
                'header_metadata': {
                    'full_path': header.full_path,
                    'parent_headers': header.parent_headers,
                    'raw_header_text': header.raw_text,
                    'header_line_number': header.line_number
                },
                'type': 'administrative' if header.administrative_info else 'markdown'
            }
            
            sections.append(section)
        
        # Handle content before first header if exists
        if header_hierarchy and header_hierarchy[0].line_number > 0:
            preamble_content = '\n'.join(lines[:header_hierarchy[0].line_number])
            if preamble_content.strip():
                preamble_section = {
                    'level': 0,
                    'title': 'Phần mở đầu',
                    'start_line': 0,
                    'end_line': header_hierarchy[0].line_number - 1,
                    'content': preamble_content,
                    'administrative_info': {},
                    'header_metadata': {
                        'full_path': 'Phần mở đầu',
                        'parent_headers': [],
                        'raw_header_text': '',
                        'header_line_number': 0
                    },
                    'type': 'preamble'
                }
                sections.insert(0, preamble_section)
        
        return sections
    
    def _extract_tables(self, content: str) -> List[Dict]:
        """Extract and parse markdown tables with header context"""
        tables = []
        table_pattern = r'\|.*?\|'
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if re.match(table_pattern, line):
                # Found start of table
                table_lines = []
                table_start = i
                
                # Collect all table lines
                while i < len(lines) and re.match(table_pattern, lines[i].strip()):
                    table_lines.append(lines[i].strip())
                    i += 1
                
                if len(table_lines) >= 2:  # At least header + separator
                    table_data = self._parse_table_lines(table_lines)
                    
                    # Find the header context for this table
                    table_header_context = self._find_table_header_context(table_start, content)
                    
                    table_info = {
                        'start_line': table_start,
                        'end_line': i - 1,
                        'raw_content': '\n'.join(table_lines),
                        'parsed_data': table_data,
                        'row_count': len(table_data.get('rows', [])),
                        'column_count': len(table_data.get('headers', [])),
                        'header_context': table_header_context  # Enhanced metadata
                    }
                    tables.append(table_info)
            else:
                i += 1
        
        return tables
    
    def _find_table_header_context(self, table_line: int, content: str) -> Dict:
        """Find the header context for a table"""
        lines = content.split('\n')
        
        # Find the most recent header before this table
        for i in range(table_line - 1, -1, -1):
            line = lines[i].strip()
            
            # Check for markdown header
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                return {
                    'type': 'markdown',
                    'level': level,
                    'title': title,
                    'line_number': i,
                    'distance_from_table': table_line - i
                }
            
            # Check for administrative header
            elif self._is_administrative_header(line):
                admin_level = self._determine_admin_level(line)
                admin_info = self._extract_administrative_info(line)
                return {
                    'type': 'administrative',
                    'level': admin_level,
                    'title': line,
                    'line_number': i,
                    'distance_from_table': table_line - i,
                    'administrative_info': admin_info
                }
        
        return {
            'type': 'unknown',
            'title': 'Không xác định được mục chứa bảng',
            'level': 0,
            'line_number': 0,
            'distance_from_table': table_line
        }
    
    def _parse_table_lines(self, lines: List[str]) -> Dict:
        """Parse markdown table lines into structured data"""
        if len(lines) < 2:
            return {}
        
        # Parse headers
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
        
        # Skip separator line (line[1])
        # Parse data rows
        rows = []
        for line in lines[2:]:
            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(cells) == len(headers):
                    rows.append(cells)
        
        return {
            'headers': headers,
            'rows': rows
        }
    
    def _extract_references(self, content: str) -> List[str]:
        """Extract internal and external references"""
        references = []
        
        # Vietnamese document reference patterns
        patterns = [
            r'(?:Điều|điều)\s+(\d+)',  # Article references
            r'(?:Khoản|khoản)\s+(\d+)',  # Clause references  
            r'(?:Chương|chương)\s+([IVX]+|\d+)',  # Chapter references
            r'(?:Phần|phần)\s+([IVX]+|\d+)',  # Section references
            r'(?:Quyết định|QĐ)\s+số\s+([\d\/\-A-Z]+)',  # Decision references
            r'(?:Thông tư|TT)\s+số\s+([\d\/\-A-Z]+)',  # Circular references
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))  # Remove duplicates
    
    def _extract_metadata(self, content: str) -> Dict:
        """Extract document metadata"""
        metadata = {}
        
        # Extract title (first header or document title pattern)
        title_match = re.search(r'^#\s+(.+)', content, re.MULTILINE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Extract document number
        doc_number_pattern = r'Số:\s*([\d\/\-A-Z]+)'
        doc_number = re.search(doc_number_pattern, content)
        if doc_number:
            metadata['document_number'] = doc_number.group(1)
        
        # Extract date
        date_pattern = r'(?:ngày|Ngày)\s+(\d{1,2})\s+(?:tháng|th)\s+(\d{1,2})\s+(?:năm|nm)\s+(\d{4})'
        date_match = re.search(date_pattern, content)
        if date_match:
            day, month, year = date_match.groups()
            metadata['date'] = f"{day}/{month}/{year}"
        
        # Extract issuing authority
        authority_patterns = [
            r'(TỔNG CÔNG TY [^\\n]+)',
            r'(BỘ [^\\n]+)',
            r'(UBND [^\\n]+)'
        ]
        
        for pattern in authority_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata['issuing_authority'] = match.group(1).strip()
                break
        
        return metadata
    
    def _extract_administrative_info(self, text: str) -> Dict:
        """Extract administrative structure information"""
        info = {}
        
        # Section info
        section_match = re.search(r'(?:PHẦN|Phần)\s+([IVX]+|[0-9]+)', text, re.IGNORECASE)
        if section_match:
            info['section'] = section_match.group(1)
            info['section_full'] = section_match.group(0)
        
        # Chapter info  
        chapter_match = re.search(r'(?:CHƯƠNG|Chương)\s+([0-9]+)', text, re.IGNORECASE)
        if chapter_match:
            info['chapter'] = chapter_match.group(1)
            info['chapter_full'] = chapter_match.group(0)
        
        # Article info
        article_match = re.search(r'(?:ĐIỀU|Điều)\s+([0-9]+)', text, re.IGNORECASE)
        if article_match:
            info['article'] = article_match.group(1)
            info['article_full'] = article_match.group(0)
        
        # Point info
        point_match = re.search(r'^([0-9]+)\.\s*(.+)', text.strip())
        if point_match:
            info['point'] = point_match.group(1)
            info['point_content'] = point_match.group(2)
        
        # Sub-point info
        subpoint_match = re.search(r'^([a-z])\)\s*(.+)', text.strip())
        if subpoint_match:
            info['subpoint'] = subpoint_match.group(1)
            info['subpoint_content'] = subpoint_match.group(2)
        
        return info
    
    def _determine_admin_level(self, text: str) -> int:
        """Determine administrative hierarchy level"""
        if re.search(r'(?:PHẦN|Phần)', text, re.IGNORECASE):
            return 1
        elif re.search(r'(?:CHƯƠNG|Chương)', text, re.IGNORECASE):
            return 2
        elif re.search(r'(?:ĐIỀU|Điều)', text, re.IGNORECASE):
            return 3
        elif re.search(r'^[0-9]+\.', text.strip()):
            return 4
        elif re.search(r'^[a-z]\)', text.strip()):
            return 5
        else:
            return 6
    
    def get_header_context_for_line(self, line_number: int, header_hierarchy: List[HeaderInfo]) -> Optional[HeaderInfo]:
        """Get the header context for a specific line number"""
        current_header = None
        
        for header in header_hierarchy:
            if header.line_number <= line_number:
                current_header = header
            else:
                break
        
        return current_header
    
    def build_full_section_path(self, section: Dict) -> str:
        """Build complete hierarchical path for a section"""
        path_parts = []
        
        # Add administrative hierarchy
        admin_info = section.get('administrative_info', {})
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
        
        # Add header metadata path if available
        header_metadata = section.get('header_metadata', {})
        if header_metadata.get('full_path') and not path_parts:
            return header_metadata['full_path']
        
        return " > ".join(path_parts) if path_parts else section.get('title', 'Unknown')