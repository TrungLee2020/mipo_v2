import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
import re
from core.config import TableConfig
import logging
from difflib import SequenceMatcher

class TableProcessor:
    """Advanced table processing for Vietnamese documents with multi-page support"""
    
    def __init__(self, config: TableConfig = None):
        self.config = config or TableConfig()
        self.logger = logging.getLogger(__name__)
    
    def process_table(self, table_data: Dict) -> Dict:
        """Process table with Vietnamese-specific enhancements and multi-page support"""
        processed_data = table_data.copy()
        
        # Extract and clean data
        headers = table_data.get('parsed_data', {}).get('headers', [])
        rows = table_data.get('parsed_data', {}).get('rows', [])
        page_info = table_data.get('page_info', {})
        
        if not headers or not rows:
            return processed_data
        
        # Detect if this is a multi-page table
        is_multipage = self._detect_multipage_table(table_data)
        
        if is_multipage:
            # Process multi-page table
            unified_data = self._process_multipage_table(table_data)
            headers = unified_data['headers']
            rows = unified_data['rows']
        
        # Clean headers
        clean_headers = self._clean_vietnamese_headers(headers)
        
        # Clean cell data
        clean_rows = self._clean_table_rows(rows)
        
        # Detect data types
        column_types = self._detect_column_types(clean_rows, clean_headers)
        
        # Create enhanced summary
        enhanced_summary = self._create_enhanced_table_summary(
            clean_headers, clean_rows, column_types, is_multipage
        )
        
        # Update processed data
        processed_data['processed_data'] = {
            'headers': clean_headers,
            'rows': clean_rows,
            'column_types': column_types,
            'enhanced_summary': enhanced_summary,
            'is_multipage': is_multipage,
            'total_pages': page_info.get('total_pages', 1) if is_multipage else 1
        }
        
        return processed_data
    
    def _detect_multipage_table(self, table_data: Dict) -> bool:
        """Detect if table spans multiple pages"""
        # Check for page information
        page_info = table_data.get('page_info', {})
        if page_info.get('total_pages', 1) > 1:
            return True
        
        # Check for continuation indicators
        rows = table_data.get('parsed_data', {}).get('rows', [])
        if not rows:
            return False
        
        # Look for continuation patterns
        continuation_patterns = [
            r'tiếp\s+theo',
            r'trang\s+\d+',
            r'phần\s+\d+',
            r'tiếp\s+trang',
            r'\.{3,}',  # Multiple dots
            r'cont[inu]*ed',
        ]
        
        first_row = ' '.join(str(cell) for cell in rows[0]).lower()
        last_row = ' '.join(str(cell) for cell in rows[-1]).lower()
        
        for pattern in continuation_patterns:
            if re.search(pattern, first_row) or re.search(pattern, last_row):
                return True
        
        # Check for incomplete data patterns
        if self._has_incomplete_data_pattern(rows):
            return True
        
        return False
    
    def _has_incomplete_data_pattern(self, rows: List[List[str]]) -> bool:
        """Check if table has patterns suggesting it's incomplete"""
        if len(rows) < 3:
            return False
        
        # Check if last few rows are incomplete
        last_rows = rows[-3:]
        incomplete_count = 0
        
        for row in last_rows:
            empty_cells = sum(1 for cell in row if not str(cell).strip())
            if empty_cells > len(row) * 0.5:  # More than 50% empty
                incomplete_count += 1
        
        return incomplete_count >= 2
    
    def _process_multipage_table(self, table_data: Dict) -> Dict:
        """Process table that spans multiple pages"""
        all_pages_data = table_data.get('all_pages_data', [])
        
        if not all_pages_data:
            # Fallback to single page data
            return {
                'headers': table_data.get('parsed_data', {}).get('headers', []),
                'rows': table_data.get('parsed_data', {}).get('rows', [])
            }
        
        unified_headers = []
        unified_rows = []
        
        # Process each page
        for page_idx, page_data in enumerate(all_pages_data):
            page_headers = page_data.get('headers', [])
            page_rows = page_data.get('rows', [])
            
            if page_idx == 0:
                # First page - establish headers
                unified_headers = page_headers
                unified_rows.extend(self._clean_page_rows(page_rows, is_first_page=True))
            else:
                # Subsequent pages
                matched_headers = self._match_headers(unified_headers, page_headers)
                if matched_headers:
                    # Headers match, merge rows
                    clean_rows = self._clean_page_rows(page_rows, is_first_page=False)
                    unified_rows.extend(clean_rows)
                else:
                    # Headers don't match, try to align columns
                    aligned_rows = self._align_columns(unified_headers, page_headers, page_rows)
                    unified_rows.extend(aligned_rows)
        
        return {
            'headers': unified_headers,
            'rows': unified_rows
        }
    
    def _match_headers(self, base_headers: List[str], page_headers: List[str]) -> bool:
        """Check if headers match between pages"""
        if len(base_headers) != len(page_headers):
            return False
        
        match_threshold = 0.8
        matches = 0
        
        for base_h, page_h in zip(base_headers, page_headers):
            similarity = SequenceMatcher(None, base_h.lower(), page_h.lower()).ratio()
            if similarity >= match_threshold:
                matches += 1
        
        return matches >= len(base_headers) * 0.7  # 70% of headers must match
    
    def _clean_page_rows(self, rows: List[List[str]], is_first_page: bool = False) -> List[List[str]]:
        """Clean rows from a specific page"""
        clean_rows = []
        
        for row_idx, row in enumerate(rows):
            # Skip header rows on subsequent pages
            if not is_first_page and row_idx == 0:
                # Check if this looks like a header row
                if self._is_likely_header_row(row):
                    continue
            
            # Skip continuation indicators
            row_text = ' '.join(str(cell) for cell in row).lower()
            if self._is_continuation_row(row_text):
                continue
            
            # Skip empty or mostly empty rows
            if self._is_empty_row(row):
                continue
            
            clean_rows.append(row)
        
        return clean_rows
    
    def _is_likely_header_row(self, row: List[str]) -> bool:
        """Check if row is likely a header row"""
        row_text = ' '.join(str(cell) for cell in row).lower()
        
        header_indicators = [
            'stt', 'số thứ tự', 'tên', 'mã số', 'số lượng',
            'ngày', 'tháng', 'năm', 'ghi chú', 'đơn vị'
        ]
        
        indicator_count = sum(1 for indicator in header_indicators 
                            if indicator in row_text)
        
        return indicator_count >= 2
    
    def _is_continuation_row(self, row_text: str) -> bool:
        """Check if row is a continuation indicator"""
        continuation_patterns = [
            r'tiếp\s+theo',
            r'trang\s+\d+',
            r'phần\s+\d+',
            r'tiếp\s+trang',
            r'^\.{3,}$',
            r'cont[inu]*ed',
            r'^\s*$'  # Empty row
        ]
        
        for pattern in continuation_patterns:
            if re.search(pattern, row_text):
                return True
        
        return False
    
    def _is_empty_row(self, row: List[str]) -> bool:
        """Check if row is empty or mostly empty"""
        non_empty_cells = sum(1 for cell in row if str(cell).strip())
        return non_empty_cells <= len(row) * 0.2  # Less than 20% filled
    
    def _align_columns(self, base_headers: List[str], page_headers: List[str], 
                      page_rows: List[List[str]]) -> List[List[str]]:
        """Align columns when headers don't match exactly"""
        aligned_rows = []
        
        if not page_headers or not page_rows:
            return aligned_rows
        
        # Create mapping between headers
        header_mapping = {}
        for i, page_header in enumerate(page_headers):
            best_match_idx = -1
            best_similarity = 0
            
            for j, base_header in enumerate(base_headers):
                similarity = SequenceMatcher(None, 
                                           page_header.lower(), 
                                           base_header.lower()).ratio()
                if similarity > best_similarity and similarity > 0.6:
                    best_similarity = similarity
                    best_match_idx = j
            
            if best_match_idx >= 0:
                header_mapping[i] = best_match_idx
        
        # Align rows based on mapping
        for row in page_rows:
            aligned_row = [''] * len(base_headers)
            
            for page_col_idx, cell in enumerate(row):
                if page_col_idx in header_mapping:
                    base_col_idx = header_mapping[page_col_idx]
                    aligned_row[base_col_idx] = cell
            
            aligned_rows.append(aligned_row)
        
        return aligned_rows
    
    def _clean_vietnamese_headers(self, headers: List[str]) -> List[str]:
        """Clean and normalize Vietnamese table headers"""
        clean_headers = []
        
        for header in headers:
            # Remove extra whitespace
            clean_header = re.sub(r'\s+', ' ', header.strip())
            
            # Remove page numbers and continuation text
            clean_header = re.sub(r'\(trang\s+\d+\)', '', clean_header, flags=re.IGNORECASE)
            clean_header = re.sub(r'\(tiếp\s+theo\)', '', clean_header, flags=re.IGNORECASE)
            
            # Normalize common Vietnamese abbreviations
            abbreviations = {
                'STT': 'Số thứ tự',
                'TT': 'Thứ tự', 
                'SL': 'Số lượng',
                'DT': 'Doanh thu',
                'ĐVHC': 'Đơn vị hành chính',
                'TCTD': 'Tổ chức tín dụng',
                'HTKT': 'Hệ thống kế toán',
                'CBCNV': 'Cán bộ công nhân viên'
            }
            
            for abbr, full in abbreviations.items():
                if clean_header.upper() == abbr:
                    clean_header = f"{full} ({abbr})"
                    break
            
            clean_headers.append(clean_header.strip())
        
        return clean_headers
    
    def _clean_table_rows(self, rows: List[List[str]]) -> List[List[str]]:
        """Clean table row data with enhanced number handling"""
        clean_rows = []
        
        for row in rows:
            clean_row = []
            for cell in row:
                # Clean cell content
                clean_cell = re.sub(r'\s+', ' ', str(cell).strip())
                
                # Handle Vietnamese number formatting
                if self._is_vietnamese_number(clean_cell):
                    clean_cell = self._normalize_vietnamese_number(clean_cell)
                
                # Handle Vietnamese currency
                if self._is_vietnamese_currency(clean_cell):
                    clean_cell = self._normalize_vietnamese_currency(clean_cell)
                
                clean_row.append(clean_cell)
            
            clean_rows.append(clean_row)
        
        return clean_rows
    
    def _is_vietnamese_number(self, text: str) -> bool:
        """Check if text is a Vietnamese formatted number"""
        # Vietnamese number patterns: 1.234.567 or 1,234,567 or mixed
        patterns = [
            r'^\d{1,3}(\.\d{3})*$',  # 1.234.567
            r'^\d{1,3}(,\d{3})*$',   # 1,234,567
            r'^\d+$',                 # Simple number
        ]
        
        for pattern in patterns:
            if re.match(pattern, text.strip()):
                return True
        return False
    
    def _normalize_vietnamese_number(self, text: str) -> str:
        """Normalize Vietnamese number format"""
        # Remove dots used as thousand separators
        normalized = text.replace('.', '')
        # Keep commas as decimal separators if they appear at the end
        if ',' in normalized and len(normalized.split(',')[-1]) <= 2:
            # This is likely a decimal separator
            parts = normalized.split(',')
            if len(parts) == 2:
                return f"{parts[0]}.{parts[1]}"
        else:
            # Remove commas used as thousand separators
            normalized = normalized.replace(',', '')
        
        return normalized
    
    def _is_vietnamese_currency(self, text: str) -> bool:
        """Check if text contains Vietnamese currency"""
        currency_patterns = [
            r'\d+.*đồng',
            r'\d+.*VND',
            r'\d+.*vnđ',
            r'đồng.*\d+',
            r'VND.*\d+',
        ]
        
        text_lower = text.lower()
        for pattern in currency_patterns:
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _normalize_vietnamese_currency(self, text: str) -> str:
        """Normalize Vietnamese currency format"""
        # Extract number part and currency unit
        number_match = re.search(r'([\d\.,]+)', text)
        if number_match:
            number_part = number_match.group(1)
            normalized_number = self._normalize_vietnamese_number(number_part)
            
            # Determine currency unit
            text_lower = text.lower()
            if 'đồng' in text_lower:
                return f"{normalized_number} đồng"
            elif 'vnd' in text_lower:
                return f"{normalized_number} VND"
            else:
                return f"{normalized_number} đồng"
        
        return text
    
    def _detect_column_types(self, rows: List[List[str]], headers: List[str]) -> Dict[str, str]:
        """Enhanced column type detection"""
        column_types = {}
        
        if not rows:
            return column_types
        
        for col_idx, header in enumerate(headers):
            column_values = []
            
            # Collect values for this column
            for row in rows:
                if col_idx < len(row):
                    column_values.append(row[col_idx])
            
            # Detect type with enhanced logic
            column_type = self._infer_column_type(column_values, header)
            column_types[header] = column_type
        
        return column_types
    
    def _infer_column_type(self, values: List[str], header: str) -> str:
        """Enhanced column type inference"""
        if not values:
            return 'text'
        
        # Check for Vietnamese-specific patterns
        header_lower = header.lower()
        
        # Sequential number (STT)
        if any(keyword in header_lower for keyword in ['stt', 'số thứ tự', 'thứ tự']):
            return 'sequence'
        
        # Date columns
        if any(keyword in header_lower for keyword in ['ngày', 'tháng', 'năm', 'thời gian', 'date']):
            return 'date'
        
        # Currency columns
        if any(keyword in header_lower for keyword in ['tiền', 'đồng', 'giá', 'phí', 'lương', 'thu nhập', 'chi phí']):
            return 'currency'
        
        # Percentage columns
        if any(keyword in header_lower for keyword in ['tỷ lệ', '%', 'phần trăm']):
            return 'percentage'
        
        # Number columns
        if any(keyword in header_lower for keyword in ['số', 'lượng', 'độ', 'chỉ số']):
            # Check if values are numeric
            numeric_count = 0
            for value in values[:min(10, len(values))]:  # Sample first 10 values
                if self._is_vietnamese_number(value.strip()) or value.strip().replace('.', '').replace(',', '').isdigit():
                    numeric_count += 1
            
            if numeric_count > len(values[:min(10, len(values))]) * 0.7:  # 70% numeric
                return 'number'
        
        # Organization/person names
        if any(keyword in header_lower for keyword in ['tên', 'họ', 'cơ quan', 'đơn vị', 'công ty', 'doanh nghiệp']):
            return 'name'
        
        # Administrative codes
        if any(keyword in header_lower for keyword in ['mã', 'số hiệu', 'quyết định', 'văn bản']):
            return 'code'
        
        # Address
        if any(keyword in header_lower for keyword in ['địa chỉ', 'nơi', 'khu vực', 'vùng']):
            return 'address'
        
        return 'text'
    
    def _create_enhanced_table_summary(self, headers: List[str], rows: List[List[str]], 
                                     column_types: Dict[str, str], is_multipage: bool = False) -> str:
        """Create enhanced summary with Vietnamese context"""
        summary_parts = []
        
        # Basic statistics
        if is_multipage:
            summary_parts.append(f"Bảng dữ liệu đa trang có {len(rows)} hàng và {len(headers)} cột")
        else:
            summary_parts.append(f"Bảng dữ liệu có {len(rows)} hàng và {len(headers)} cột")
        
        # Column descriptions
        summary_parts.append("\n**Mô tả các cột:**")
        for header in headers:
            col_type = column_types.get(header, 'text')
            type_description = self._get_type_description(col_type)
            summary_parts.append(f"- {header}: {type_description}")
        
        # Data insights
        insights = self._generate_table_insights(headers, rows, column_types)
        if insights:
            summary_parts.append(f"\n**Thông tin quan trọng:**")
            summary_parts.extend([f"- {insight}" for insight in insights])
        
        # Multi-page specific insights
        if is_multipage:
            summary_parts.append(f"\n**Lưu ý:** Bảng này được ghép từ nhiều trang, đã được xử lý tự động để loại bỏ header trùng lặp và căn chỉnh dữ liệu.")
        
        return '\n'.join(summary_parts)
    
    def _get_type_description(self, col_type: str) -> str:
        """Get Vietnamese description for column type"""
        descriptions = {
            'text': 'Dữ liệu văn bản',
            'number': 'Dữ liệu số',
            'currency': 'Dữ liệu tiền tệ',
            'date': 'Dữ liệu ngày tháng',
            'name': 'Tên người/tổ chức',
            'code': 'Mã số/ký hiệu',
            'sequence': 'Số thứ tự',
            'percentage': 'Tỷ lệ phần trăm',
            'address': 'Địa chỉ/Vị trí'
        }
        return descriptions.get(col_type, 'Dữ liệu văn bản')
    
    def _generate_table_insights(self, headers: List[str], rows: List[List[str]], 
                               column_types: Dict[str, str]) -> List[str]:
        """Generate enhanced insights about table data"""
        insights = []
        
        # Find key columns
        key_columns = []
        for header in headers:
            header_lower = header.lower()
            if any(keyword in header_lower for keyword in ['tên', 'số', 'mã', 'loại', 'đơn vị']):
                key_columns.append(header)
        
        if key_columns:
            insights.append(f"Các cột quan trọng: {', '.join(key_columns)}")
        
        # Numeric insights
        for header, col_type in column_types.items():
            if col_type in ['number', 'currency']:
                col_idx = headers.index(header)
                values = []
                
                for row in rows:
                    if col_idx < len(row):
                        try:
                            # Try to convert to number
                            val_str = row[col_idx].replace(',', '').replace('.', '')
                            # Handle currency symbols
                            val_str = re.sub(r'[^\d]', '', val_str)
                            if val_str.isdigit():
                                values.append(int(val_str))
                        except:
                            continue
                
                if values and len(values) > 1:
                    min_val = min(values)
                    max_val = max(values)
                    avg_val = sum(values) // len(values)
                    
                    if col_type == 'currency':
                        insights.append(f"{header}: từ {min_val:,} đến {max_val:,} đồng (TB: {avg_val:,} đồng)")
                    else:
                        insights.append(f"{header}: từ {min_val:,} đến {max_val:,} (TB: {avg_val:,})")
        
        # Data completeness
        total_cells = len(headers) * len(rows)
        empty_cells = 0
        
        for row in rows:
            for cell in row:
                if not str(cell).strip():
                    empty_cells += 1
        
        if total_cells > 0:
            completeness = ((total_cells - empty_cells) / total_cells) * 100
            insights.append(f"Độ đầy đủ dữ liệu: {completeness:.1f}%")
        
        return insights